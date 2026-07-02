"""
Pardarshita - Train Ticket Booking & Queue Management System

A Streamlit-based web application demonstrating:
- Relational database design with SQLite and ACID transactions
- FIFO waitlist queue management with automatic seat promotion
- Role-based access control (admin / user) with SHA-256 password hashing
- ML-powered waitlist confirmation probability using Logistic Regression
- Split-route booking via SQL self-join queries
"""

import streamlit as st
import sqlite3
import random
import pandas as pd
import hashlib

try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    LogisticRegression = None

# ============================================================
# DATABASE CONNECTION HELPER
# ============================================================

def get_db_connection():
    """Connect to database. Foreign keys help keep data safe."""
    conn = sqlite3.connect('railway.db')
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def hash_password(password):
    """Create a deterministic hash for password storage and verification."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database():
    """Create tables if they don't exist, add default trains."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trains (
                train_no INTEGER PRIMARY KEY,
                train_name TEXT NOT NULL,
                source TEXT NOT NULL,
                dest TEXT NOT NULL,
                seats_total INTEGER NOT NULL,
                seats_avail INTEGER NOT NULL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bookings (
                pnr TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                train_no INTEGER NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (train_no) REFERENCES trains(train_no)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS waitlist (
                token_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                train_no INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (train_no) REFERENCES trains(train_no)
            )
        ''')

        # Store AI predictions and eventual outcomes for waitlist confirmations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS waitlist_predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                train_no INTEGER NOT NULL,
                queue_position INTEGER NOT NULL,
                seats_total INTEGER NOT NULL,
                seats_avail_at_booking INTEGER NOT NULL,
                predicted_probability REAL NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence_band TEXT NOT NULL,
                model_used TEXT NOT NULL,
                actual_confirmed INTEGER DEFAULT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved_at DATETIME DEFAULT NULL,
                FOREIGN KEY (train_no) REFERENCES trains(train_no)
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_waitlist_predictions_train
            ON waitlist_predictions (train_no)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_waitlist_predictions_token
            ON waitlist_predictions (token_id)
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                is_active INTEGER NOT NULL DEFAULT 1
            )
        ''')

        cursor.execute("SELECT COUNT(*) FROM trains WHERE train_no = 101")
        if cursor.fetchone()[0] == 0:
            cursor.execute('''
                INSERT INTO trains (train_no, train_name, source, dest, seats_total, seats_avail)
                VALUES (101, 'Shatabdi Express', 'Delhi', 'Bhopal', 50, 2)
            ''')
            cursor.execute('''
                INSERT OR IGNORE INTO trains VALUES (train_no, train_name, source, dest, seats_total, seats_avail)
                VALUES (102, 'Rajdhani Express', 'Mumbai', 'Kolkata', 60, 10)
            ''')

        cursor.execute('''
            INSERT OR IGNORE INTO trains (train_no, train_name, source, dest, seats_total, seats_avail)
            VALUES (201, 'Bundelkhand Link', 'Delhi', 'Agra', 40, 12)
        ''')
        cursor.execute('''
            INSERT OR IGNORE INTO trains (train_no, train_name, source, dest, seats_total, seats_avail)
            VALUES (202, 'Malwa Connector', 'Agra', 'Bhopal', 40, 14)
        ''')

        cursor.execute('''
            INSERT OR IGNORE INTO users (username, password_hash, role, is_active)
            VALUES (?, ?, ?, ?)
        ''', ('admin', hash_password('admin123'), 'admin', 1))
        cursor.execute('''
            INSERT OR IGNORE INTO users (username, password_hash, role, is_active)
            VALUES (?, ?, ?, ?)
        ''', ('user', hash_password('user123'), 'user', 1))

        conn.commit()
        return True, "Database ready"

    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        return False, str(e)

    finally:
        if conn:
            conn.close()


def authenticate_user(username, password):
    """Validate login credentials and return role information."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT username, role
            FROM users
            WHERE username = ?
              AND password_hash = ?
              AND is_active = 1
        ''', (username, hash_password(password)))

        user_row = cursor.fetchone()
        if not user_row:
            return False, "Invalid username or password"

        username_db, role = user_row
        return True, {
            'username': username_db,
            'role': role
        }

    except sqlite3.Error as e:
        return False, str(e)

    finally:
        if conn:
            conn.close()


# ============================================================
# BACKEND FUNCTIONS
# ============================================================

def get_all_trains():
    """Get list of all trains from database."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trains")
        trains = cursor.fetchall()
        return True, trains
    except sqlite3.Error as e:
        return False, str(e)
    finally:
        if conn:
            conn.close()


def _clamp(value, minimum, maximum):
    """Clamp a numeric value inside [minimum, maximum]."""
    return max(minimum, min(maximum, value))


def predict_waitlist_confirmation(cursor, train_no, queue_position, seats_total, seats_avail):
    """Estimate chance that a waitlist entry gets confirmed."""
    cursor.execute('''
        SELECT queue_position, seats_total, seats_avail_at_booking, actual_confirmed
        FROM waitlist_predictions
        WHERE actual_confirmed IS NOT NULL
    ''')
    training_rows = cursor.fetchall()

    cursor.execute('''
        SELECT AVG(actual_confirmed), COUNT(*)
        FROM waitlist_predictions
        WHERE train_no = ?
          AND actual_confirmed IS NOT NULL
    ''', (train_no,))
    train_history_rate, train_history_count = cursor.fetchone()

    cursor.execute('''
        SELECT AVG(actual_confirmed)
        FROM waitlist_predictions
        WHERE actual_confirmed IS NOT NULL
    ''')
    global_history_rate = cursor.fetchone()[0]

    probability = None
    model_used = "heuristic"

    has_model_data = len(training_rows) >= 12
    class_values = set([row[3] for row in training_rows])

    if LogisticRegression and has_model_data and len(class_values) >= 2:
        try:
            x_train = [[row[0], row[1], row[2]] for row in training_rows]
            y_train = [row[3] for row in training_rows]

            model = LogisticRegression(max_iter=500)
            model.fit(x_train, y_train)
            probability = float(model.predict_proba([[queue_position, seats_total, seats_avail]])[0][1])
            model_used = "logistic_regression"
        except Exception:
            probability = None

    if probability is None:
        baseline = 0.78 - ((queue_position - 1) * 0.13)
        capacity_factor = min(0.10, seats_total / 1000.0)
        availability_factor = 0.0
        if seats_total > 0:
            availability_factor = (seats_avail / seats_total) * 0.08

        history_reference = 0.5
        if train_history_rate is not None and train_history_count >= 3:
            history_reference = train_history_rate
        elif global_history_rate is not None:
            history_reference = global_history_rate

        history_factor = (history_reference - 0.5) * 0.35
        probability = baseline + capacity_factor + availability_factor + history_factor

    probability = _clamp(probability, 0.03, 0.97)

    if probability >= 0.70:
        label = "High chance"
    elif probability >= 0.40:
        label = "Moderate chance"
    else:
        label = "Low chance"

    distance = abs(probability - 0.5)
    if distance >= 0.25:
        confidence_band = "High"
    elif distance >= 0.12:
        confidence_band = "Medium"
    else:
        confidence_band = "Low"

    return {
        'probability': round(probability, 4),
        'probability_percent': round(probability * 100, 1),
        'label': label,
        'confidence_band': confidence_band,
        'model_used': model_used,
        'training_samples': len(training_rows)
    }


def save_waitlist_prediction(cursor, token_id, name, train_no, queue_position, seats_total, seats_avail, prediction):
    """Persist waitlist prediction details for later evaluation."""
    cursor.execute('''
        INSERT INTO waitlist_predictions (
            token_id,
            name,
            train_no,
            queue_position,
            seats_total,
            seats_avail_at_booking,
            predicted_probability,
            predicted_label,
            confidence_band,
            model_used
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        token_id,
        name,
        train_no,
        queue_position,
        seats_total,
        seats_avail,
        prediction['probability'],
        prediction['label'],
        prediction['confidence_band'],
        prediction['model_used']
    ))


def mark_waitlist_prediction_confirmed(cursor, token_id, train_no):
    """Mark the associated prediction as confirmed when waitlist entry is promoted."""
    cursor.execute('''
        UPDATE waitlist_predictions
        SET actual_confirmed = 1,
            resolved_at = CURRENT_TIMESTAMP
        WHERE prediction_id = (
            SELECT prediction_id
            FROM waitlist_predictions
            WHERE token_id = ?
              AND train_no = ?
              AND actual_confirmed IS NULL
            ORDER BY prediction_id DESC
            LIMIT 1
        )
    ''', (token_id, train_no))


def get_train_details(train_no):
    """Get one train's details by train number."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT train_no, train_name, source, dest, seats_total, seats_avail
            FROM trains
            WHERE train_no = ?
        ''', (train_no,))
        row = cursor.fetchone()

        if not row:
            return False, "Train not found"

        return True, {
            'train_no': row[0],
            'train_name': row[1],
            'source': row[2],
            'dest': row[3],
            'seats_total': row[4],
            'seats_avail': row[5]
        }
    except sqlite3.Error as e:
        return False, str(e)
    finally:
        if conn:
            conn.close()


def find_split_routes(source, dest, exclude_train_no=None, limit=5):
    """Find A->C and C->B options where both legs have available seats."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = '''
            SELECT
                t1.train_no,
                t1.train_name,
                t1.source,
                t1.dest,
                t1.seats_avail,
                t2.train_no,
                t2.train_name,
                t2.source,
                t2.dest,
                t2.seats_avail
            FROM trains t1
            INNER JOIN trains t2 ON t1.dest = t2.source
            WHERE t1.source = ?
              AND t2.dest = ?
              AND t1.seats_avail > 0
              AND t2.seats_avail > 0
              AND t1.train_no != t2.train_no
        '''
        params = [source, dest]

        if exclude_train_no is not None:
            query += " AND t1.train_no != ? AND t2.train_no != ?"
            params.extend([exclude_train_no, exclude_train_no])

        query += '''
            ORDER BY
                CASE WHEN t1.seats_avail < t2.seats_avail THEN t1.seats_avail ELSE t2.seats_avail END DESC,
                t1.train_no ASC,
                t2.train_no ASC
            LIMIT ?
        '''
        params.append(limit)

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        split_routes = []
        for row in rows:
            split_routes.append({
                'leg1_train_no': row[0],
                'leg1_train_name': row[1],
                'leg1_source': row[2],
                'transfer_city': row[3],
                'leg1_seats': row[4],
                'leg2_train_no': row[5],
                'leg2_train_name': row[6],
                'leg2_source': row[7],
                'leg2_dest': row[8],
                'leg2_seats': row[9],
                'guaranteed_seats': min(row[4], row[9])
            })

        return True, split_routes
    except sqlite3.Error as e:
        return False, str(e)
    finally:
        if conn:
            conn.close()


def book_split_route(name, first_train_no, second_train_no):
    """Book both legs atomically for a split route journey."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT train_no, train_name, source, dest, seats_avail
            FROM trains
            WHERE train_no = ?
        ''', (first_train_no,))
        leg1 = cursor.fetchone()

        cursor.execute('''
            SELECT train_no, train_name, source, dest, seats_avail
            FROM trains
            WHERE train_no = ?
        ''', (second_train_no,))
        leg2 = cursor.fetchone()

        if not leg1 or not leg2:
            return False, "One or both split-route trains were not found"

        if leg1[3] != leg2[2]:
            return False, "Selected trains are not a valid split route"

        if leg1[4] <= 0 or leg2[4] <= 0:
            return False, "One leg is full now. Please retry with another split route"

        pnr_leg1 = "PNR" + str(random.randint(10000000, 99999999))
        pnr_leg2 = "PNR" + str(random.randint(10000000, 99999999))

        cursor.execute('''
            INSERT INTO bookings (pnr, name, train_no, status)
            VALUES (?, ?, ?, ?)
        ''', (pnr_leg1, name, first_train_no, 'CONFIRMED'))

        cursor.execute('''
            INSERT INTO bookings (pnr, name, train_no, status)
            VALUES (?, ?, ?, ?)
        ''', (pnr_leg2, name, second_train_no, 'CONFIRMED'))

        cursor.execute('''
            UPDATE trains
            SET seats_avail = seats_avail - 1
            WHERE train_no = ?
        ''', (first_train_no,))

        cursor.execute('''
            UPDATE trains
            SET seats_avail = seats_avail - 1
            WHERE train_no = ?
        ''', (second_train_no,))

        conn.commit()

        return True, {
            'name': name,
            'transfer_city': leg1[3],
            'leg1': {
                'pnr': pnr_leg1,
                'train_no': leg1[0],
                'train_name': leg1[1],
                'source': leg1[2],
                'dest': leg1[3]
            },
            'leg2': {
                'pnr': pnr_leg2,
                'train_no': leg2[0],
                'train_name': leg2[1],
                'source': leg2[2],
                'dest': leg2[3]
            }
        }

    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        return False, str(e)

    finally:
        if conn:
            conn.close()


def book_ticket(name, train_no):
    """Book a ticket. If full, add to waitlist."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT seats_avail, seats_total, train_name FROM trains WHERE train_no = ?", (train_no,))
        result = cursor.fetchone()

        if not result:
            return False, "Train not found"

        seats_avail, seats_total, train_name = result

        if seats_avail > 0:
            pnr = "PNR" + str(random.randint(10000000, 99999999))

                cursor.execute('''
                INSERT INTO bookings (pnr, name, train_no, status)
                VALUES (?, ?, ?, ?)
            ''', (pnr, name, train_no, 'CONFIRMED'))

                cursor.execute('''
                UPDATE trains
                SET seats_avail = seats_avail - 1
                WHERE train_no = ?
            ''', (train_no,))

            conn.commit()

            return True, {
                'type': 'booking',
                'pnr': pnr,
                'name': name,
                'train_name': train_name,
                'train_no': train_no,
                'status': 'CONFIRMED'
            }

        else:
            cursor.execute('''
                INSERT INTO waitlist (name, train_no)
                VALUES (?, ?)
            ''', (name, train_no))

            token_id = cursor.lastrowid

            cursor.execute('''
                SELECT COUNT(*) FROM waitlist WHERE train_no = ?
            ''', (train_no,))
            queue_position = cursor.fetchone()[0]

            prediction = predict_waitlist_confirmation(
                cursor,
                train_no,
                queue_position,
                seats_total,
                seats_avail
            )
            save_waitlist_prediction(
                cursor,
                token_id,
                name,
                train_no,
                queue_position,
                seats_total,
                seats_avail,
                prediction
            )

            conn.commit()

            return True, {
                'type': 'waitlist',
                'token_id': token_id,
                'name': name,
                'train_name': train_name,
                'train_no': train_no,
                'queue_position': queue_position,
                'prediction': prediction
            }

    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        return False, str(e)

    finally:
        if conn:
            conn.close()


def cancel_ticket(pnr):
    """Cancel a booking. Promote first person from waitlist if any."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT name, train_no FROM bookings WHERE pnr = ?", (pnr,))
        result = cursor.fetchone()

        if not result:
            return False, "PNR not found"

        cancelled_name, train_no = result

        cursor.execute("DELETE FROM bookings WHERE pnr = ?", (pnr,))

        cursor.execute('''
            SELECT token_id, name
            FROM waitlist
            WHERE train_no = ?
            ORDER BY token_id ASC
            LIMIT 1
        ''', (train_no,))

        waitlist_entry = cursor.fetchone()
        promoted_passenger = None

        if waitlist_entry:
            token_id, waitlist_name = waitlist_entry
            promoted_pnr = "PNR" + str(random.randint(10000000, 99999999))

            cursor.execute('''
                INSERT INTO bookings (pnr, name, train_no, status)
                VALUES (?, ?, ?, ?)
            ''', (promoted_pnr, waitlist_name, train_no, 'CONFIRMED'))

            mark_waitlist_prediction_confirmed(cursor, token_id, train_no)

            cursor.execute('DELETE FROM waitlist WHERE token_id = ?', (token_id,))

            promoted_passenger = {
                'name': waitlist_name,
                'pnr': promoted_pnr,
                'token_id': token_id
            }
        else:
            cursor.execute('''
                UPDATE trains
                SET seats_avail = seats_avail + 1
                WHERE train_no = ?
            ''', (train_no,))

        conn.commit()

        return True, {
            'pnr': pnr,
            'name': cancelled_name,
            'train_no': train_no,
            'promoted_passenger': promoted_passenger
        }

    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        return False, str(e)

    finally:
        if conn:
            conn.close()


def check_pnr(pnr):
    """Look up a booking using PNR."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT 
                b.pnr,
                b.name,
                b.status,
                t.train_no,
                t.train_name,
                t.source,
                t.dest
            FROM bookings b
            INNER JOIN trains t ON b.train_no = t.train_no
            WHERE b.pnr = ?
        ''', (pnr,))

        result = cursor.fetchone()

        if not result:
            return False, "PNR not found"

        pnr, name, status, train_no, train_name, source, dest = result

        return True, {
            'pnr': pnr,
            'name': name,
            'status': status,
            'train_no': train_no,
            'train_name': train_name,
            'source': source,
            'dest': dest
        }

    except sqlite3.Error as e:
        return False, str(e)

    finally:
        if conn:
            conn.close()


def add_train(train_no, train_name, source, dest, seats_total):
    """Add a new train."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT train_no FROM trains WHERE train_no = ?", (train_no,))
        if cursor.fetchone():
            return False, "Train already exists"

        cursor.execute('''
            INSERT INTO trains (train_no, train_name, source, dest, seats_total, seats_avail)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (train_no, train_name, source, dest, seats_total, seats_total))

        conn.commit()
        return True, "Train added successfully!"

    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        return False, str(e)

    finally:
        if conn:
            conn.close()


def delete_train(train_no):
    """Delete a train."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT train_no FROM trains WHERE train_no = ?", (train_no,))
        if not cursor.fetchone():
            return False, "Train not found"

        cursor.execute("DELETE FROM trains WHERE train_no = ?", (train_no,))

        conn.commit()
        return True, "Train deleted successfully!"

    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        return False, str(e)

    finally:
        if conn:
            conn.close()


def reset_seats(train_no, seats):
    """Set available seats on a train with capacity validation."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT seats_total
            FROM trains
            WHERE train_no = ?
        ''', (train_no,))
        train_row = cursor.fetchone()

        if not train_row:
            return False, "Train not found"

        seats_total = train_row[0]
        if seats < 0 or seats > seats_total:
            return False, "Available seats must be between 0 and " + str(seats_total)

        cursor.execute('''
            UPDATE trains
            SET seats_avail = ?
            WHERE train_no = ?
        ''', (seats, train_no))

        conn.commit()
        return True, "Available seats updated for train " + str(train_no)

    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        return False, str(e)

    finally:
        if conn:
            conn.close()


def reset_train_to_full(train_no):
    """Reset one train's available seats to its total capacity."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT seats_total
            FROM trains
            WHERE train_no = ?
        ''', (train_no,))
        train_row = cursor.fetchone()

        if not train_row:
            return False, "Train not found"

        seats_total = train_row[0]
        cursor.execute('''
            UPDATE trains
            SET seats_avail = ?
            WHERE train_no = ?
        ''', (seats_total, train_no))

        conn.commit()
        return True, "Train " + str(train_no) + " reset to full capacity (" + str(seats_total) + ")"

    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        return False, str(e)

    finally:
        if conn:
            conn.close()


def reset_all_trains_to_full():
    """Reset all trains' available seats to full capacity."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM trains")
        train_count = cursor.fetchone()[0]

        if train_count == 0:
            return False, "No trains available"

        cursor.execute('''
            UPDATE trains
            SET seats_avail = seats_total
        ''')

        conn.commit()
        return True, "All " + str(train_count) + " trains reset to full capacity"

    except sqlite3.Error as e:
        if conn:
            conn.rollback()
        return False, str(e)

    finally:
        if conn:
            conn.close()


def get_waitlist(train_no):
    """Get waitlist for a train."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT token_id, name, timestamp
            FROM waitlist
            WHERE train_no = ?
            ORDER BY token_id
        ''', (train_no,))

        waitlist = cursor.fetchall()
        return True, waitlist

    except sqlite3.Error as e:
        return False, str(e)

    finally:
        if conn:
            conn.close()


def get_all_bookings(train_no):
    """Get all bookings for a train."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT pnr, name, status
            FROM bookings
            WHERE train_no = ?
            ORDER BY pnr DESC
        ''', (train_no,))

        bookings = cursor.fetchall()
        return True, bookings

    except sqlite3.Error as e:
        return False, str(e)

    finally:
        if conn:
            conn.close()


def get_train_live_summary():
    """Get live counts and seat state for every train."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                t.train_no,
                t.train_name,
                t.source,
                t.dest,
                t.seats_total,
                t.seats_avail,
                (SELECT COUNT(*) FROM bookings b WHERE b.train_no = t.train_no) AS confirmed_count,
                (SELECT COUNT(*) FROM waitlist w WHERE w.train_no = t.train_no) AS waitlist_count
            FROM trains t
            ORDER BY t.train_no ASC
        ''')

        rows = cursor.fetchall()
        return True, rows

    except sqlite3.Error as e:
        return False, str(e)

    finally:
        if conn:
            conn.close()


def get_prediction_history(train_no, limit=15):
    """Get recent AI waitlist predictions for a train."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                token_id,
                name,
                queue_position,
                predicted_probability,
                predicted_label,
                confidence_band,
                model_used,
                actual_confirmed,
                created_at,
                resolved_at
            FROM waitlist_predictions
            WHERE train_no = ?
            ORDER BY prediction_id DESC
            LIMIT ?
        ''', (train_no, limit))
        rows = cursor.fetchall()
        return True, rows
    except sqlite3.Error as e:
        return False, str(e)
    finally:
        if conn:
            conn.close()


def display_booking_result(result):
    """Render booking or waitlist response in a consistent format."""
    if result['type'] == 'booking':
        st.success("Booking Confirmed!")
        st.balloons()
        st.write("PNR Number: " + result['pnr'])
        st.write("Name: " + result['name'])
        st.write("Train: " + result['train_name'] + " (" + str(result['train_no']) + ")")
        st.write("Status: " + result['status'])
        st.warning("Save your PNR: " + result['pnr'])
    else:
        st.warning("Train is Full - Added to Waitlist")
        st.write("Token ID: " + str(result['token_id']))
        st.write("Queue Position: " + str(result['queue_position']))
        st.write("You will get a ticket if someone cancels (FIFO order)")

        prediction = result.get('prediction')
        if prediction:
            st.info(
                "AI Prediction: " + prediction['label'] +
                " (" + str(prediction['probability_percent']) + "% chance of confirmation)"
            )


# ============================================================
# STREAMLIT WEB INTERFACE
# ============================================================

st.set_page_config(page_title="Pardarshita", page_icon="📋", layout="wide")

if 'db_initialized' not in st.session_state:
    init_database()
    st.session_state.db_initialized = True
    st.rerun()

if 'auth_user' not in st.session_state:
    st.session_state.auth_user = None

if 'auth_role' not in st.session_state:
    st.session_state.auth_role = None

if 'split_route_context' not in st.session_state:
    st.session_state.split_route_context = None

st.markdown("---")
st.title("Pardarshita: Transparent Reservation & Queue Management System")
st.markdown("A simple system to book train tickets with transparent queue management")
st.markdown("---")

st.sidebar.title("Login")

if not st.session_state.auth_user or not st.session_state.auth_role:
    with st.sidebar.form("login_form"):
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        login_clicked = st.form_submit_button("Login")

    st.sidebar.caption("Demo users: admin/admin123 and user/user123")

    if login_clicked:
        if not username_input.strip() or not password_input:
            st.sidebar.error("Please enter username and password")
        else:
            login_success, login_result = authenticate_user(username_input.strip(), password_input)
            if login_success:
                st.session_state.auth_user = login_result['username']
                st.session_state.auth_role = login_result['role']
                st.rerun()
            else:
                st.sidebar.error("Login failed: " + login_result)

    st.info("Please login from the sidebar to use the system")
    st.stop()

current_user = st.session_state.auth_user
current_role = st.session_state.auth_role

st.sidebar.success("Logged in as: " + current_user + " (" + current_role.upper() + ")")
if st.sidebar.button("Logout"):
    st.session_state.auth_user = None
    st.session_state.auth_role = None
    st.session_state.split_route_context = None
    st.rerun()

st.sidebar.title("Menu")
if current_role == 'admin':
    menu_options = ["Home", "View Trains", "Live Data", "Admin"]
else:
    menu_options = ["Home", "View Trains", "Book Ticket", "Check PNR", "Cancel Ticket", "View Waitlist"]

page = st.sidebar.radio("Select an option:", menu_options)

# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":
    st.header("Welcome to Pardarshita")
    st.markdown("""
    ### About This App
    This is a transparent reservation system. It shows how to:
    - Save bookings to a database
    - Handle waiting lists fairly (FIFO)
    - Keep data safe with ACID transactions
    
    **Features:**
    - View all trains with available seats
    - Book tickets and get a PNR number
    - Check your booking status anytime
    - Cancel tickets (waitlist passengers get promoted)
    - See the waitlist queue
    
    Start by clicking a menu option on the left!
    """)

# ============================================================
# VIEW TRAINS PAGE
# ============================================================

elif page == "View Trains":
    st.header("Available Trains")

    success, data = get_all_trains()

    if success:
        if len(data) == 0:
            st.warning("No trains available")
        else:
            df = pd.DataFrame(data, columns=[
                'Train Number', 'Train Name', 'From', 'To', 'Total Seats', 'Available Seats'
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.error("Error: Could not load trains")

# ============================================================
# BOOK TICKET PAGE
# ============================================================

elif page == "Book Ticket":
    st.header("Book a Ticket")

    with st.form("book_form", clear_on_submit=True):
        name = st.text_input("Your Name")
        train_no = st.number_input("Train Number", min_value=1, value=101)
        use_split_routing = st.checkbox("Try split routing if direct train is full", value=True)
        submit_booking = st.form_submit_button("Book Ticket")

    if submit_booking:
        if not name:
            st.error("Please enter your name")
            st.session_state.split_route_context = None
        else:
            train_no = int(train_no)
            detail_success, train_detail = get_train_details(train_no)

            if not detail_success:
                st.error("Error: " + train_detail)
                st.session_state.split_route_context = None
            elif train_detail['seats_avail'] > 0:
                st.session_state.split_route_context = None
                success, result = book_ticket(name, train_no)
                if success:
                    display_booking_result(result)
                else:
                    st.error("Error: " + result)
            else:
                if use_split_routing:
                    split_success, split_routes = find_split_routes(
                        train_detail['source'],
                        train_detail['dest'],
                        exclude_train_no=train_no
                    )

                    if split_success and split_routes:
                        st.session_state.split_route_context = {
                            'name': name,
                            'direct_train_no': train_no,
                            'direct_train_name': train_detail['train_name'],
                            'source': train_detail['source'],
                            'dest': train_detail['dest'],
                            'routes': split_routes
                        }
                        st.warning("Direct train is full. Split routing suggestions are available below.")
                    elif not split_success:
                        st.error("Could not search split routes: " + split_routes)
                        st.session_state.split_route_context = None
                    else:
                        st.info("No split route found. Proceeding with direct train waitlist.")
                        st.session_state.split_route_context = None
                        success, result = book_ticket(name, train_no)
                        if success:
                            display_booking_result(result)
                        else:
                            st.error("Error: " + result)
                else:
                    st.session_state.split_route_context = None
                    success, result = book_ticket(name, train_no)
                    if success:
                        display_booking_result(result)
                    else:
                        st.error("Error: " + result)

    split_context = st.session_state.split_route_context
    if split_context:
        st.markdown("---")
        st.subheader("Split Routing Suggestions")
        st.write(
            "Direct train " + split_context['direct_train_name'] +
            " (" + str(split_context['direct_train_no']) + ") is full for route " +
            split_context['source'] + " to " + split_context['dest'] + "."
        )

        route_labels = []
        for route in split_context['routes']:
            route_labels.append(
                str(route['leg1_train_no']) + " " + route['leg1_train_name'] +
                " (" + route['leg1_source'] + "->" + route['transfer_city'] + ") + " +
                str(route['leg2_train_no']) + " " + route['leg2_train_name'] +
                " (" + route['leg2_source'] + "->" + route['leg2_dest'] + ") | Seats: " +
                str(route['guaranteed_seats'])
            )

        selected_label = st.selectbox(
            "Choose a split route:",
            options=route_labels,
            key="split_route_selection"
        )
        selected_route = split_context['routes'][route_labels.index(selected_label)]

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Book Selected Split Route", key="book_split_route_btn"):
                split_book_success, split_result = book_split_route(
                    split_context['name'],
                    selected_route['leg1_train_no'],
                    selected_route['leg2_train_no']
                )

                if split_book_success:
                    st.success("Split route booking confirmed!")
                    st.balloons()
                    st.write("Passenger: " + split_result['name'])
                    st.write(
                        "Leg 1: " + split_result['leg1']['train_name'] +
                        " (" + str(split_result['leg1']['train_no']) + ") - PNR " +
                        split_result['leg1']['pnr']
                    )
                    st.write(
                        "Leg 2: " + split_result['leg2']['train_name'] +
                        " (" + str(split_result['leg2']['train_no']) + ") - PNR " +
                        split_result['leg2']['pnr']
                    )
                    st.warning(
                        "Save both PNRs: " + split_result['leg1']['pnr'] +
                        " and " + split_result['leg2']['pnr']
                    )
                    st.session_state.split_route_context = None
                else:
                    st.error("Split route booking failed: " + split_result)

        with col2:
            if st.button("Join Direct Waitlist", key="join_waitlist_btn"):
                wait_success, wait_result = book_ticket(
                    split_context['name'],
                    split_context['direct_train_no']
                )

                if wait_success:
                    display_booking_result(wait_result)
                    st.session_state.split_route_context = None
                else:
                    st.error("Error: " + wait_result)

        with col3:
            if st.button("Dismiss Suggestions", key="dismiss_split_btn"):
                st.session_state.split_route_context = None
                st.rerun()

# ============================================================
# CHECK PNR PAGE
# ============================================================

elif page == "Check PNR":
    st.header("Check PNR Status")

    with st.form("check_form"):
        pnr = st.text_input("Enter PNR Number").strip().upper()

        if st.form_submit_button("Check"):
            if not pnr:
                st.error("Please enter PNR")
            else:
                success, result = check_pnr(pnr)

                if success:
                    st.success("Booking Found!")
                    st.write("PNR: " + result['pnr'])
                    st.write("Name: " + result['name'])
                    st.write("Train: " + result['train_name'] + " (" + str(result['train_no']) + ")")
                    st.write("Route: " + result['source'] + " to " + result['dest'])
                    st.write("Status: " + result['status'])
                else:
                    st.error("Error: " + result)

# ============================================================
# CANCEL TICKET PAGE
# ============================================================

elif page == "Cancel Ticket":
    st.header("Cancel Ticket")

    with st.form("cancel_form"):
        pnr = st.text_input("Enter PNR Number").strip().upper()
        confirm = st.checkbox("I want to cancel this ticket")

        if st.form_submit_button("Cancel"):
            if not pnr:
                st.error("Please enter PNR")
            elif not confirm:
                st.error("Please confirm cancellation")
            else:
                success, result = cancel_ticket(pnr)

                if success:
                    st.success("Ticket Cancelled!")
                    st.write("PNR: " + result['pnr'])
                    st.write("Name: " + result['name'])

                    if result.get('promoted_passenger'):
                        promoted = result['promoted_passenger']
                        st.success("Waitlist passenger automatically promoted!")
                        st.balloons()
                        st.write("Promoted Passenger: " + promoted['name'])
                        st.write("New PNR: " + promoted['pnr'])
                    else:
                        st.info("Refund will be processed in 5-7 business days")
                else:
                    st.error("Error: " + result)

# ============================================================
# LIVE DATA PAGE
# ============================================================

elif page == "Live Data":
    if current_role != 'admin':
        st.error("Access denied. Admin role required.")
    else:
        st.header("Live Data View")
        st.markdown("Train-wise live bookings, waitlist, and seat status")

        col_refresh, col_note = st.columns([1, 3])
        with col_refresh:
            if st.button("Refresh Live Data"):
                st.rerun()
        with col_note:
            st.caption(
                "Updates are automatic after booking, cancellation, split-route booking, "
                "seat reset, and add/delete train."
            )

        summary_success, summary_data = get_train_live_summary()

        if not summary_success:
            st.error("Error loading train summary: " + summary_data)
        elif len(summary_data) == 0:
            st.info("No trains available")
        else:
            st.subheader("All Trains - Live Summary")
            df_summary = pd.DataFrame(summary_data, columns=[
                'Train Number', 'Train Name', 'From', 'To', 'Total Seats',
                'Available Seats', 'Confirmed Bookings', 'Waitlist Count'
            ])
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Detailed Data For Each Train")

            for row in summary_data:
                train_no, train_name, source, dest, seats_total, seats_avail, confirmed_count, waitlist_count = row
                header = (
                    str(train_no) + " - " + train_name + " (" + source + " -> " + dest + ") | " +
                    "Seats: " + str(seats_avail) + "/" + str(seats_total) +
                    " | Bookings: " + str(confirmed_count) +
                    " | Waitlist: " + str(waitlist_count)
                )

                with st.expander(header, expanded=False):
                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:
                        st.write("Confirmed Bookings")
                        success_b, bookings = get_all_bookings(train_no)
                        if success_b and bookings:
                            df_bookings = pd.DataFrame(bookings, columns=['PNR', 'Name', 'Status'])
                            st.dataframe(df_bookings, use_container_width=True, hide_index=True)
                        elif success_b:
                            st.info("No bookings for this train")
                        else:
                            st.error("Could not load bookings")

                    with detail_col2:
                        st.write("Waiting List")
                        success_w, waitlist = get_waitlist(train_no)
                        if success_w and waitlist:
                            df_waitlist = pd.DataFrame(waitlist, columns=['Token ID', 'Name', 'Joined At'])
                            st.dataframe(df_waitlist, use_container_width=True, hide_index=True)
                        elif success_w:
                            st.info("No waitlist for this train")
                        else:
                            st.error("Could not load waitlist")

                    st.write("AI Waitlist Prediction History")
                    success_p, prediction_rows = get_prediction_history(train_no)
                    if success_p and prediction_rows:
                        prediction_data = []
                        for pred in prediction_rows:
                            token_id, pred_name, queue_pos, pred_prob, pred_label, conf_band, model_used, actual_confirmed, created_at, resolved_at = pred
                            outcome = "Pending"
                            if actual_confirmed == 1:
                                outcome = "Confirmed"

                            prediction_data.append({
                                'Token ID': token_id,
                                'Name': pred_name,
                                'Queue Position': queue_pos,
                                'Predicted Chance %': round(float(pred_prob) * 100, 1),
                                'Prediction': pred_label,
                                'Confidence': conf_band,
                                'Model': model_used,
                                'Outcome': outcome,
                                'Predicted At': created_at,
                                'Resolved At': resolved_at
                            })

                        df_predictions = pd.DataFrame(prediction_data)
                        st.dataframe(df_predictions, use_container_width=True, hide_index=True)
                    elif success_p:
                        st.info("No prediction history for this train yet")
                    else:
                        st.error("Could not load prediction history")

# ============================================================
# VIEW WAITLIST PAGE
# ============================================================

elif page == "View Waitlist":
    st.header("Waiting List (FIFO Queue)")

    train_no = st.number_input("Train Number", min_value=1, value=101)

    if st.button("Show Waitlist"):
        success, data = get_waitlist(train_no)

        if success:
            if len(data) == 0:
                st.info("No one is waiting - all seats available!")
            else:
                df = pd.DataFrame(data, columns=['Token ID', 'Name', 'Joined At'])
                st.warning(str(len(df)) + " people waiting (in queue order):")
                st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.error("Error: " + data)

# ============================================================
# ADMIN PAGE
# ============================================================

elif page == "Admin":
    if current_role != 'admin':
        st.error("Access denied. Admin role required.")
    else:
        st.header("Admin Panel")

        st.info("Manage train seats")

        summary_success, summary_data = get_train_live_summary()
        if summary_success and summary_data:
            st.subheader("Train Snapshot")
            df_admin_summary = pd.DataFrame(summary_data, columns=[
                'Train Number', 'Train Name', 'From', 'To', 'Total Seats',
                'Available Seats', 'Confirmed Bookings', 'Waitlist Count'
            ])
            st.dataframe(df_admin_summary, use_container_width=True, hide_index=True)
        elif not summary_success:
            st.error("Error loading train snapshot: " + summary_data)

        st.subheader("Add New Train")
        with st.form("add_train_form"):
            col1, col2 = st.columns(2)

            with col1:
                new_train_no = st.number_input("Train Number", min_value=1, value=103)
                new_train_name = st.text_input("Train Name", value="Express Train")

            with col2:
                new_source = st.text_input("Source City", value="City A")
                new_dest = st.text_input("Destination City", value="City B")

            new_seats = st.number_input("Total Seats", min_value=1, value=50)

            if st.form_submit_button("Add Train"):
                success, message = add_train(new_train_no, new_train_name, new_source, new_dest, new_seats)

                if success:
                    st.success(message)
                else:
                    st.error("Error: " + message)

        st.markdown("---")

        st.subheader("Reset Available Seats")

        if summary_success and summary_data:
            train_label_to_row = {
                str(row[0]) + " - " + row[1] + " (" + row[2] + " -> " + row[3] + ")": row
                for row in summary_data
            }
            train_labels = list(train_label_to_row.keys())

            with st.form("set_available_seats_form"):
                selected_train_label = st.selectbox("Select Train", options=train_labels)
                selected_train = train_label_to_row[selected_train_label]

                selected_train_no = int(selected_train[0])
                selected_train_total = int(selected_train[4])
                selected_train_avail = int(selected_train[5])

                st.caption(
                    "Current available seats: " + str(selected_train_avail) +
                    "/" + str(selected_train_total)
                )

                new_available_seats = st.number_input(
                    "Set New Available Seats",
                    min_value=0,
                    max_value=selected_train_total,
                    value=selected_train_avail
                )

                if st.form_submit_button("Update Available Seats"):
                    success, message = reset_seats(selected_train_no, int(new_available_seats))

                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error("Error: " + message)

            with st.form("reset_train_full_form"):
                selected_full_label = st.selectbox(
                    "Select Train to Reset to Full Capacity",
                    options=train_labels,
                    key="reset_full_train_select"
                )
                confirm_full_reset = st.checkbox("I want to reset this train to full capacity")

                if st.form_submit_button("Reset Selected Train To Full"):
                    if not confirm_full_reset:
                        st.error("Please confirm reset")
                    else:
                        train_no_to_full = int(train_label_to_row[selected_full_label][0])
                        success, message = reset_train_to_full(train_no_to_full)

                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error("Error: " + message)

            with st.form("reset_all_full_form"):
                confirm_reset_all = st.checkbox("I want to reset all trains to full capacity")

                if st.form_submit_button("Reset All Trains To Full"):
                    if not confirm_reset_all:
                        st.error("Please confirm reset")
                    else:
                        success, message = reset_all_trains_to_full()

                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error("Error: " + message)
        else:
            st.info("No trains found to reset")

        st.markdown("---")

        st.subheader("Delete Train")
        with st.form("delete_train_form"):
            delete_train_no = st.number_input("Train Number to Delete", min_value=1, value=103)
            confirm_delete = st.checkbox("I want to delete this train")

            if st.form_submit_button("Delete Train"):
                if not confirm_delete:
                    st.error("Please confirm deletion")
                else:
                    success, message = delete_train(delete_train_no)

                    if success:
                        st.success(message)
                    else:
                        st.error("Error: " + message)


st.markdown("---")
st.markdown("Pardarshita - Demonstrating fair, transparent reservation systems with FIFO queue management")