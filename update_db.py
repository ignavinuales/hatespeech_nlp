from datetime import datetime
from dotenv import load_dotenv
import oracledb
import os


class UpdateDB():
    """Class to represent a new user input in the database"""

    def __init__(self, input: str, output: str):
        """Constructs the inital attributes for the UpdateDB class.
        
        Args:
            input [str]: text input from the user to be fed in the NLP model.
            output [str]: output from the NLP model.
        """

        self.input = input
        self.output = output
        self._current_time = datetime.utcnow()   # Get current time in utc timezone

    def _connect_db(self):
        """Connect to Oracle database and create a cursor object"""

        # Get credentials to connect to Oracle Autonomous Database
        load_dotenv('.env')
        user = os.getenv("secretUser")
        key = os.getenv("secretKey")
        dsn = os.getenv("secretDsn")

        # Connect to Oracle db
        self._conn = oracledb.connect(
            user=user,
            password=key,
            dsn=dsn)
        self.cursor = self._conn.cursor()

    def _disconnect_db(self) -> None:
        """Disconnect from the databae and close the cursor object"""

        self.cursor.close()
        self._conn.close()

    def update_db(self) -> None:
        """Updates database. It connects to the db, executes a sql statement, commits, and close the connection."""

        self._connect_db()

        try:
            self.cursor.execute(self.sql_command, self.sql_command_values)
        except oracledb.Error as error:
            self._conn.rollback()
            print('Error:'.format(error))
        else:
            self._conn.commit()
            self._disconnect_db()

    def new_entry_db(self):
        """New entry to the database. It inserts new input text from the user and the output from the ML model. """

        self.sql_command = "INSERT INTO model_monitoring (input_text, output, execution_datetime) VALUES (:input_text, :output, :execution_datetime)"
        self.sql_command_values = [self.input, self.output, self._current_time]
        self.update_db()  # Update DB

    def add_feedback_db(self, feedback:str) -> None:
        """It adds the feedback from the user to the feedback column in the database."""

        self.sql_command = "UPDATE model_monitoring SET feedback = :feedback WHERE execution_datetime =:execution_datetime"
        self.sql_command_values = [feedback, self._current_time]
        self.update_db()  # Update DB
