import asyncio
import os
import threading
import pandas as pd

from abc import abstractmethod, ABC
from SeedCore.Core.Delegate import Delegate
from SeedCore.TagSystem.Tag import TagContainer

"""
세션 기본 클래스, Expire 및 기본적인 세션 관련 기능 지원.
서브시스템에 의해서 라이프사이클이 관리 되는 오브젝트들의 베이스 클래스로 사용 될 수 있습니다.
"""

class SessionBase(ABC):

    def __init__(self):
        self.start_timestamp = pd.Timestamp.now()
        self.refreshed_timestamp = self.start_timestamp
        self.timer_task = None

        self.timeout_seconds = 300  # Default timeout duration in seconds (5 minutes)

        self.on_expire = Delegate()

        self.session_tags : TagContainer = TagContainer()

    def __del__(self):
        self.on_end_session()

    def __eq__(self, other):
        return self.get_session_id() == other.get_session_id()

    def set_timeout_seconds(self, timeout_seconds):
        """
        Sets the timeout duration for the session in seconds.
        This is used to determine when the session should expire.
        """
        self.timeout_seconds = timeout_seconds

    def set_timer(self, timeout_seconds=-1):
        """
        Sets or resets an asyncio timer task that will call `expire` after the given timeout.
        """
        self.clear_timer()

        timeout = timeout_seconds if timeout_seconds >= 0 else self.timeout_seconds

        async def timer_coroutine():
            try:
                await asyncio.sleep(timeout)
                self.expire()
            except asyncio.CancelledError:
                pass  # Timer was cleared before it finished

        self.timer_task = asyncio.create_task(timer_coroutine())

    def clear_timer(self):
        """
        Cancels any existing asyncio timer task.
        """
        if self.timer_task and not self.timer_task.done():
            self.timer_task.cancel()
        self.timer_task = None

    def refresh_timer(self, timeout_seconds=-1):
        """
        Resets the timer and updates the timestamp.
        """
        self.refreshed_timestamp = pd.Timestamp.now()
        self.set_timer(timeout_seconds if timeout_seconds >= 0 else self.timeout_seconds)

    def expire(self):
        """
        Expires the session, saving logs and performing cleanup.
        """
        self.clear_timer()  # Clear any existing timer

        # Trigger the on_expire event
        self.on_expire.broadcast(self)

        self.on_end_session()

    def get_session_data_path(self):
        """
        Returns the relative path for the session data.
        """
        return os.path.join("saved", "session_data", self.get_session_id())

    @abstractmethod
    def get_session_id(self):
        """
        Returns a unique session identifier based on the start timestamp.
        Anything that can be hashed can be used as a session ID.
        """
        return "NULL"


    #Bubbled up events for session lifecycle management
    def on_begin_session(self):
        """
        Called when a session begins. Can be overridden to add custom behavior.
        """

    #Bubbled up events for session lifecycle management
    def on_end_session(self):
        """
        Called when a session ends. Can be overridden to add custom behavior.
        """
        self.clear_timer()

    def get_session_tag_container(self) -> TagContainer:
        return self.session_tags