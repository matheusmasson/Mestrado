import sqlite3
import os.path

def select_all_triples():
    conn = sqlite3.connect("/home/matheus/Documents/Dissertacao-Dev/deepmusicdb.db")

    cursor = conn.cursor()
    sql = "SELECT userid, musicid, count, train, test, cold FROM 'main'.'user_music'"
    cursor.execute(sql)
    rows = cursor.fetchall()

    return rows

def select_all_triples_small():
    conn = sqlite3.connect("/home/matheus/Documents/Dissertacao-Dev/deepmusicdb.db")

    cursor = conn.cursor()
    sql = "SELECT userid, musicid, count, train, test, cold FROM 'main'.'user_music' where hard = 0"
    cursor.execute(sql)
    rows = cursor.fetchall()

    return rows

def select_all_triples_full():
    conn = sqlite3.connect("/home/matheus/Documents/Dissertacao-Dev/deepmusicdb_full.db")

    cursor = conn.cursor()
    sql = "SELECT userid, musicid, count, train, test, cold FROM 'main'.'user_music'"
    cursor.execute(sql)
    rows = cursor.fetchall()

    return rows


def select_all_rtings_train():
    conn = sqlite3.connect("/home/matheus/Documents/Dissertacao-Dev/deepmusicdb.db")

    cursor = conn.cursor()
    sql = "SELECT musicid, sum(count) FROM 'main'.'user_music' where train = 1 group by musicid order by musicid"
    cursor.execute(sql)
    rows = cursor.fetchall()

    return rows

def select_all_rtings_train_full():
    conn = sqlite3.connect("/home/matheus/Documents/Dissertacao-Dev/deepmusicdb_full.db")

    cursor = conn.cursor()
    sql = "SELECT musicid, sum(count) FROM 'main'.'user_music' where train = 1 group by musicid order by musicid"
    cursor.execute(sql)
    rows = cursor.fetchall()

    return rows


def select_music_factor_by_id(id):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "deepmusicdb.db")
    conn = sqlite3.connect(db_path)
    conn.text_factory = str

    cursor = conn.cursor()

    sql = "SELECT factor FROM music WHERE id=?"
    cursor.execute(sql, [(id)])
    row = cursor.fetchone()
    if row is not None:
        return row[0]

    return None


def select_music_factor_by_id_train(id):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "deepmusicdb.db")
    conn = sqlite3.connect(db_path)
    conn.text_factory = str

    cursor = conn.cursor()

    sql = "SELECT factor FROM music a WHERE id=? and train = 1"
    cursor.execute(sql, [(id)])
    row = cursor.fetchone()
    if row is not None:
        return row[0]

    return None

def select_songs_file_names():
    conn = sqlite3.connect("/home/matheus/Documents/Dissertacao-Dev/deepmusicdb.db")
    cursor = conn.cursor()

    sql = "SELECT id, mapid FROM music  where downloaded = 1 order by id"
    cursor.execute(sql)
    rows = cursor.fetchall()

    return rows

def select_songs_file_names_small():
    conn = sqlite3.connect("/home/matheus/Documents/Dissertacao-Dev/deepmusicdb.db")
    cursor = conn.cursor()

    sql = "SELECT id, mapid FROM music where id in (select mapid from music_p) order by id"
    cursor.execute(sql)
    rows = cursor.fetchall()

    return rows

def select_songs_file_names_ids(ids):
    conn = sqlite3.connect("/home/matheus/Documents/Dissertacao-Dev/deepmusicdb.db")
    cursor = conn.cursor()

    sql = "SELECT id, mapid FROM music where id in (" + ids + ") order by id"


    cursor.execute(sql)
    rows = cursor.fetchall()

    return rows

def select_test_cold_song_ids_full():
    conn = sqlite3.connect("/home/matheus/Documents/Dissertacao-Dev/deepmusicdb_full.db")
    cursor = conn.cursor()

    #sql = "SELECT id FROM music where (test = 1) or (cold = 1)"
    sql = "SELECT id FROM music where (test = 1) and (hard = 0)"
    cursor.execute(sql)
    rows = cursor.fetchall()

    return rows

def select_songs_teste():
    conn = sqlite3.connect("/home/matheus/Documents/Dissertacao-Dev/deepmusicdb_full.db")
    cursor = conn.cursor()

    sql = "SELECT id, title, artist, mapid FROM music where id in (3192, 29044, 1038, 3405, 25420, 3418, 16107, 5913)"
    cursor.execute(sql)
    rows = cursor.fetchall()

    return rows

def select_test_cold_song_ids_small():
    conn = sqlite3.connect("/home/matheus/Documents/Dissertacao-Dev/deepmusicdb.db")
    cursor = conn.cursor()

    sql = "select id from music where (test = 1) or (cold = 1)"
    cursor.execute(sql)
    rows = cursor.fetchall()

    return rows


def select_triple_by_user_id(userid):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "deepmusicdb.db")
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()
    sql = "SELECT userid, musicid, count FROM user_music WHERE userid=? order by count desc"
    cursor.execute(sql, [(userid)])
    rows = cursor.fetchall()

    return rows

def select_music_order_by_count():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "deepmusicdb.db")
    conn = sqlite3.connect(db_path)
    conn.text_factory = str

    cursor = conn.cursor()

    sql = "SELECT id FROM music WHERE downloaded = 1 order by totalcount desc"
    cursor.execute(sql)
    rows = cursor.fetchall()

    return rows


def select_triple_user(userid):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "deepmusicdb.db")
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()
    sql = "SELECT userid, musicid, count FROM user_music WHERE userid=? order by count desc"
    cursor.execute(sql, [(userid)])
    rows = cursor.fetchall()

    return rows

def select_triple_user_test(userid):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "deepmusicdb.db")
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()
    sql = "SELECT userid, musicid, count FROM user_music inner join music on user_music.musicid = music.id where user_music.userid = ? and user_music.test = 1 and music.train = 1"
    cursor.execute(sql, [(userid)])
    rows = cursor.fetchall()

    return rows

def select_n_users_top(num):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(BASE_DIR, "deepmusicdb.db")
    conn = sqlite3.connect(db_path)
    conn.text_factory = str

    cursor = conn.cursor()

    #sql = "select userid, count(*)  as n from user_music group by userid order by n desc LIMIT ?"
    sql = "select id from user order by id LIMIT ?"
    cursor.execute(sql, [(num)])
    rows = cursor.fetchall()

    return rows

def select_triples_train():
    conn = sqlite3.connect("/home/matheus/Documents/Dissertacao-Dev/deepmusicdb.db")

    cursor = conn.cursor()
    sql = "SELECT userid, musicid, count FROM 'main'.'user_music' where train = 1"
    cursor.execute(sql)
    rows = cursor.fetchall()

    return rows

def select_triples_train_small():
    conn = sqlite3.connect("/home/matheus/Documents/Dissertacao-Dev/deepmusicdb.db")

    cursor = conn.cursor()
    sql = "SELECT userid, musicid, count FROM 'main'.'user_music' where train = 1"
    cursor.execute(sql)
    rows = cursor.fetchall()

    return rows

