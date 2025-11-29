import asyncio
import json
import os
from asyncio import Lock

from SeedCore.SharedFunctionLibrary import LOG_TEXT
from SeedCore.Session.SessionBase import SessionBase
from SeedCore.SharedFunctionLibrary import get_cwd_root_path

"""
Singleton Subsystem for managing configuration instances.
You can retrieve a configuration instance by providing the configuration file name.
"""

class ConfigSubsystem:
    _instance = None
    _locks: dict[str, Lock] = {}  # Session ID to Lock mapping
    _global_lock = Lock()  # For creating new locks

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigSubsystem, cls).__new__(cls)
            cls._instance.sessions = {}
            cls._instance._locks = {}  # Initialize locks dict
            cls._instance._global_lock = Lock()  # Initialize global lock
        return cls._instance

    @classmethod
    def get(cls) -> 'ConfigSubsystem':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_config(cls, config_full_path) -> 'ConfigInstance':

        if not cls.get().has_session(config_full_path):
            config = ConfigInstance(config_full_path)
            cls.get().assign_session(config)
            return config

        return cls.get().get_session(config_full_path)

    @classmethod
    def get_config_from_path_parts(cls, *config_path_parts) -> 'ConfigInstance':
        """
        concat the config path parts to form the config file name
        """
        config_file_name = os.path.join(get_cwd_root_path(), *config_path_parts)
        return cls.get_config(config_file_name)


    def assign_session(self, session):
        session_id = session.get_session_id()

        if session_id in self.sessions:
            LOG_TEXT(f"Config session with ID {session_id} already exists. Returning existing session.", colorKey='GREEN', verbosity='INFO')

            return self.sessions[session_id]

        session.on_expire.add_callback(lambda s: self.on_session_expired(session))
        self.sessions[session_id] = session

        LOG_TEXT(f"Created new config session with ID {session_id}.", colorKey='GREEN',verbosity='INFO')

        return session

    async def aassign_session(self, session):
        """
        Asynchronously assign a session. Thread-safe with per-session locks.
        """
        session_id = session.get_session_id()

        async with self._global_lock:
            if str(session_id) not in self._locks:
                self._locks[str(session_id)] = Lock()

        async with self._locks[str(session_id)]:
            if session_id in self.sessions:
                LOG_TEXT(f"Config session with ID {session_id} already exists. Returning existing session.", colorKey='GREEN',
                         verbosity='INFO')
                return self.sessions[session_id]

            session.on_expire.add_callback(lambda s: self.on_session_expired(session))
            self.sessions[session_id] = session
            LOG_TEXT(f"Created new config session with ID {session_id}.", colorKey='GREEN', verbosity='INFO')
            return session

    def has_session(self, session_id):
        return session_id in self.sessions

    async def ahas_session(self, session_id):
        """
        Asynchronously check if session exists.
        Waits if lock exists but doesn't acquire it.
        """
        try:
            # Check if lock exists and wait if it does
            if str(session_id) in self._locks:
                # Wait for any ongoing operations to complete
                while self._locks[str(session_id)].locked():
                    await asyncio.sleep(0.1)

            # Then check session existence
            return session_id in self.sessions

        except Exception as e:
            LOG_TEXT(f"Error checking config session {session_id}: {e}", colorKey='RED', verbosity='ERROR')
            return False

    def get_session(self, session_id):
        if not self.has_session(session_id):
            raise ValueError(f"Config session with ID {session_id} does not exist. Please create it first.")
            return None
        return self.sessions[session_id]

    async def aget_session(self, session_id):
        """
        Asynchronously get a session. Thread-safe with per-session locks.
        """
        if str(session_id) in self._locks:
            # Wait for any ongoing operations to complete
            while self._locks[str(session_id)].locked():
                await asyncio.sleep(0.1)

        # Then check session existence
        return self.sessions[session_id]

    def end_all_sessions(self):
        for session_key, session in self.sessions.items():
            print(f"ending config session {session_key}...")
            session.on_end_session()

        self.sessions.clear()


    # callback for session expiration
    def on_session_expired(self, session):
        print(f"Config session {session.get_session_id()} has expired. Ending session...")

        del self.sessions[session.get_session_id()]

        print(f"Config session {session.get_session_id()} has been removed from active sessions.")




"""
This class defines a class `ConfigInstance` that loads a configuration file in JSON format and provides a method to retrieve values by key.
Loaded configuration is stored in the ConfigSubsystem singleton instance, with the key being the configuration file name.
"""

class ConfigInstance(SessionBase):
    def __init__(self, config_full_path):
        SessionBase.__init__(self)
        self.config_full_path = config_full_path
        self.config = self.load_config(self.config_full_path)

    def get_session_id(self):
        return self.config_full_path

    @staticmethod
    def load_config(config_full_path):

        # try loading the config file with various encodings
        encodings_to_try = ['utf-8', 'utf-16', 'latin-1', 'utf-8-sig']

        for encoding in encodings_to_try:
            try:
                with open(config_full_path, 'r', encoding=encoding) as f:
                    return json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

    def get(self, key):
        return self.config.get(key)

