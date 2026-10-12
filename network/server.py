from typing import Optional

from data_server import NavigationDataServer


nav_server: Optional[NavigationDataServer] = None


def start_tcp_server():
    global nav_server

    nav_server = NavigationDataServer(
        host="0.0.0.0",
        port=8765,
        video_port=8766,
    )

    nav_server.start()

    return nav_server