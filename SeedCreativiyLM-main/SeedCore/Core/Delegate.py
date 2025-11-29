from typing import Callable, List


class Delegate:
    """세션 이벤트 델리게이트 클래스"""

    def __init__(self):
        self._callbacks: List[Callable] = []

    def add_callback(self, callback: Callable):
        """콜백 함수 추가"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """콜백 함수 제거"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def clear_callbacks(self):
        """모든 콜백 함수 제거"""
        self._callbacks.clear()

    def broadcast(self, *args, **kwargs):
        """모든 등록된 콜백 함수 호출"""
        results = []
        for callback in self._callbacks:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error in callback {callback.__name__}: {e}")
                results.append(None)
        return results

    def __len__(self):
        return len(self._callbacks)