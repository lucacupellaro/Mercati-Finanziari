from ibapi.client import EClient
from ibapi.wrapper import EWrapper
import time

class TestClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId: int, errorCode: int, errorString: str, advancedOrderRejectJson=""):
        print(f"Errore: reqId={reqId} errorCode={errorCode} errorString={errorString}")

    def nextValidId(self, orderId: int):
        print(f"Connessione stabilita. Prossimo Order ID: {orderId}")
        self.disconnect()

def main():
    client = TestClient()
    client.connect("127.0.0.1", 7497, 0)  # Indirizzo IP e porta predefiniti per TWS API (porta Paper Trading)

    # Esegui la loop di elaborazione degli eventi per gestire le risposte dell'API
    client.run()

if __name__ == "__main__":
    main()