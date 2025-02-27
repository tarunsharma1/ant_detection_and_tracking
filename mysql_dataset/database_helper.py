import mysql.connector
from mysql.connector import Error

def create_connection(host_name, user_name, user_password, db_name=None):
    connection = None
    try:
        if db_name:
            connection = mysql.connector.connect(
                host=host_name,
                user=user_name,
                password=user_password,
                database=db_name
            )
        else:
            connection = mysql.connector.connect(
                host=host_name,
                user=user_name,
                password=user_password
            )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection


def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        #print("Query successful")
        return cursor.fetchall() 
    except Error as e:
        print(f"The error '{e}' occurred")


def create_tables(connection):
	### Create tables
	create_sites_table = """
	CREATE TABLE IF NOT EXISTS Sites (
	    site_id INT PRIMARY KEY,
	    site_name VARCHAR(255) NOT NULL,
	    location VARCHAR(255) NOT NULL
	);
	"""

	create_videos_table = """
	CREATE TABLE IF NOT EXISTS Videos (
	    video_id VARCHAR(255) PRIMARY KEY,
	    site_id INT,
	    time_stamp DATETIME NOT NULL,
	    temperature FLOAT,
	    humidity FLOAT,
	    LUX FLOAT,
	    FOREIGN KEY (site_id) REFERENCES Sites(site_id)
	);
	"""

	create_counts_table = """
	CREATE TABLE IF NOT EXISTS Counts (
	    video_id VARCHAR(255) PRIMARY KEY,
	    blob_detection_average_count FLOAT,
	    blob_detection_std_dev FLOAT,
	    yolo FLOAT,
	    yolo_std_dev FLOAT,
	    herdnet FLOAT,
	    FOREIGN KEY (video_id) REFERENCES Videos(video_id)
	);
	"""

	execute_query(connection, create_sites_table)
	execute_query(connection, create_videos_table)
	execute_query(connection, create_counts_table)
	connection.commit()


def insert_site(connection, site_id, site_name, location):
    query = f"""
    INSERT INTO Sites (site_id, site_name, location)
    VALUES ({site_id}, '{site_name}', '{location}');
    """
    execute_query(connection, query)
    connection.commit()

def insert_video(connection, video_id, site_id, time_stamp, temperature, humidity, lux):
    query = f"""
    INSERT INTO Videos (video_id, site_id, time_stamp, temperature, humidity, LUX)
    VALUES ('{video_id}', {site_id}, '{time_stamp}', {temperature}, {humidity}, {lux});
    """
    execute_query(connection, query)
    connection.commit()

def add_video_to_counts_table(connection, video_id):
	query = f"""INSERT INTO Counts (video_id) VALUES ('{video_id}');"""
	execute_query(connection, query)
	connection.commit()

def get_site_id_from_site_name(connection, site_name):
	query = f"""
	SELECT site_id from Sites where site_name = '{site_name}';
	"""
	value = execute_query(connection, query)
	return value[0][0]

