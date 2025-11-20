from spider.client import SpiderClient

def main() -> None:
    config_path = "config/test-remote-online-distill.yaml"
    with SpiderClient.from_config(config_path) as client:
        submission = client.submit_job()
        status = client.poll_job(submission["job_id"], interval=5.0, wait_for_completion=True)

        if status["status"] == "completed":
            client.download_result(submission["job_id"], destination="artifacts/test-remote.json")
        else:
            raise RuntimeError(status.get("error") or status.get("messages"))

if __name__ == "__main__":
    main()