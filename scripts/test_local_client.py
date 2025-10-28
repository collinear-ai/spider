from spider.client import SpiderClient

def main() -> None:
    with SpiderClient.from_config("config/test-local.yaml") as client:
        submission = client.submit_job()
        job_id = submission["job_id"]
        print(f"Job submitted: {job_id}")

        status = client.poll_job(job_id, interval=5.0, timeout=600)
        print(f"Final status: {status['status']}")

        if status["status"] == "completed":
            artifact_path = client.download_result(job_id, destination="artifacts/test-local.jsonl")
            print(f"Artifacts saved to {artifact_path}")
        else:
            print("Messages: ", status.get("messages", []))
            if status.get("error"):
                print("Error: ", status["error"])

if __name__ == "__main__":
    main()