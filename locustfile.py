from locust import HttpUser, task, between


class YoloV7(HttpUser):
    wait_time = between(0, 1)

    @task(2)
    def index(self):
        url = f"{self.host}/yolov7/test"
        print(url)
        response = self.client.post(url=url, json={"imgDir": "/app/inference/images"})
        assert response.status_code == 200

