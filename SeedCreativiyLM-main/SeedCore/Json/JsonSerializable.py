from abc import abstractmethod

class JsonSerializable:

    """
    Base class for JSON serializable objects.
    json 시리얼라이제이션에 필요한 기본적인 기능과 가상 함수들을 정의해 놓는 인터페이스 격의 클래스입니다.
    """

    @classmethod
    @abstractmethod
    def make_from_json(cls, json_data):
        """
        Serialize a DownloadableTable instance from JSON data.
        :param json_data: The JSON data to create the instance from.
        """
        raise NotImplementedError(f"Subclass {cls.__name__} must implement an abstract method - make_from_json")

    @classmethod
    @abstractmethod
    def export_as_json(cls, in_object):
        """
        Export the DownloadableTable data to a JSON-compatible dictionary.
        :return: A dictionary representation of the DownloadableTable.
        """
        raise NotImplementedError(f"Subclass {cls.__name__} must implement an abstract method - export_as_json")
