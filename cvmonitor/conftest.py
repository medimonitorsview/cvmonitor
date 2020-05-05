import pytest

@pytest.fixture(scope='session')
def app(request):
    from .server import Server
    server = Server()
    return server.app
