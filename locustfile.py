from locust import HttpUser, task, between


class YoloV7(HttpUser):
    @task(2)
    def index(self):
        url = f"{self.host}/yolov7/test"
        url = f"{self.host}/api/test"
        print(url)
        # response = self.client.post(url=url, json={"imgDir": "/Users/lixiaolong/manyProjects/githubProjects/yolov7-on-nvidia-orin/inference/objects"})
        response = self.client.post(url=url, json={"imgDir": "/app/inference/objects"})
        assert response.status_code == 200
        print(response.content.decode())

