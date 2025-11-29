import asyncio

from asyncio import Lock

class SessionSubsystem:
    _instance = None
    _locks: dict[str, Lock] = {}  # Session ID to Lock mapping
    _global_lock = Lock()  # For creating new locks

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionSubsystem, cls).__new__(cls)
            cls._instance.sessions = {}
            cls._instance._locks = {}  # Initialize locks dict
            cls._instance._global_lock = Lock()  # Initialize global lock
        return cls._instance

    @classmethod
    def get(cls) -> 'SessionSubsystem':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


    # async control
    async def _wait_lock_for(self, session_id: str):
        if session_id in self._locks:
            # Wait for any ongoing operations to complete
            while self._locks[session_id].locked():
                await asyncio.sleep(0.05)

    async def _get_lock_for(self, session_id: str):
        async with self._global_lock:
            if session_id not in self._locks:
                self._locks[session_id] = Lock()



    def assign_session(self, session):
        session_id = session.get_session_id()

        if session_id in self.sessions:
            return self.sessions[session_id]

        session.on_expire.add_callback(lambda s: self.on_session_expired(session))
        self.sessions[session_id] = session

        return session

    async def aassign_session(self, session):
        """
        Asynchronously assign a session. Thread-safe with per-session locks.
        """
        session_id = session.get_session_id()

        await self._get_lock_for(str(session_id))

        async with self._locks[str(session_id)]:
            if session_id in self.sessions:
                return self.sessions[session_id]

            session.on_expire.add_callback(lambda s: self.on_session_expired(session))
            self.sessions[session_id] = session
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
            await self._wait_lock_for(str(session_id))

            # Then check session existence
            return session_id in self.sessions

        except Exception as e:
            return False

    def get_session(self, session_id):
        if not self.has_session(session_id):
            return None
        return self.sessions[session_id]

    async def aget_session(self, session_id):
        """
        Asynchronously get a session. Thread-safe with per-session locks.
        """
        await self._wait_lock_for(str(session_id))

        # Then check session existence
        return self.sessions[session_id]

    def end_all_sessions(self):
        for session_key, session in self.sessions.items():
            session.on_end_session()

        self.sessions.clear()

    # callback for session expiration
    def on_session_expired(self, session):
        del self.sessions[session.get_session_id()]

