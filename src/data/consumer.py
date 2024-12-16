# src/data/consumer.py
import pika
import json
from pymongo import MongoClient
from ..utils.config import Config

class IoTDataConsumer:
    def __init__(self):
        # RabbitMQ setup
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=Config.RABBITMQ_HOST,
                port=Config.RABBITMQ_PORT
            )
        )
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='iot_data')

        # MongoDB setup
        self.client = MongoClient(Config.MONGODB_URI)
        self.db = self.client.iot_database
        self.collection = self.db.raw_data

    def callback(self, ch, method, properties, body):
        """Process received messages"""
        data = json.loads(body)
        print(f"Received data: {data}")
        # Store in MongoDB
        self.collection.insert_one(data)

    def start_consuming(self):
        """Start consuming messages"""
        self.channel.basic_consume(
            queue='iot_data',
            on_message_callback=self.callback,
            auto_ack=True
        )
        print("Started consuming messages...")
        self.channel.start_consuming()

if __name__ == "__main__":
    consumer = IoTDataConsumer()
    consumer.start_consuming()