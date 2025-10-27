import os
import time

import requests

# --- Configuration ---
# IMPORTANT: Replace this with your actual Ballchasing API key.
# You can create one here: https://ballchasing.com/upload
with open(os.path.expanduser("~/ballchasing_api")) as file:
    AUTH_TOKEN = file.read().strip()

# The rank you want to search for.
# Examples: "Diamond I", "Champion III", "Grand Champion II", "Supersonic Legend"
TARGET_RANK = "grand-champion-1"

# The number of replays you want to download.
REPLAY_COUNT = 2

# The directory where you want to save the replay files.
DOWNLOAD_DIR = "replays"
# --- End of Configuration ---

BASE_API_URL = "https://ballchasing.com/api/"


def fetch_replay_list(rank, count):
    """
    Fetches a list of replays from the Ballchasing API based on rank.

    Args:
        rank (str): The Rocket League rank to filter by (e.g., "Diamond I").
        count (int): The maximum number of replays to fetch.

    Returns:
        list: A list of replay data dictionaries, or an empty list if an error occurs.
    """
    if AUTH_TOKEN == "YOUR_AUTH_TOKEN_HERE":
        print("ERROR: Please replace 'YOUR_AUTH_TOKEN_HERE' with your actual Ballchasing API key.")
        return []

    headers = {"Authorization": AUTH_TOKEN}
    # We use 'ranked-duels' for 1v1 replays.
    params = {
        "playlist": "ranked-duels",
        "min-rank": rank,
        "count": count,
        "sort-by": "replay-date",
        "sort-dir": "desc",
    }

    print(f"Fetching {count} replay(s) for rank: '{rank}'...")

    try:
        response = requests.get(f"{BASE_API_URL}replays", headers=headers, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            if "list" in data and data["list"]:
                print(f"Successfully found {len(data['list'])} replays.")
                return data["list"]
            else:
                print("No replays found matching the criteria.")
                return []
        else:
            print(f"Error fetching replays. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}")
        return []


def download_replay(replay_info, directory):
    """
    Downloads a single replay file from its URL.

    Args:
        replay_info (dict): A dictionary containing the replay's metadata, including 'id' and 'link'.
        directory (str): The directory to save the file in.
    """
    replay_id = replay_info.get("id")
    download_url = f"{BASE_API_URL}replays/{replay_id}/file"

    if not replay_id or not download_url:
        print("Skipping replay due to missing 'id' or 'link'.")
        return

    file_path = os.path.join(directory, f"{replay_id}.replay")

    if os.path.exists(file_path):
        print(f"Replay {replay_id} already exists. Skipping.")
        return

    print(f"Downloading replay '{replay_id}'...")
    try:
        # The API documentation recommends adding the Authorization header for downloads too
        headers = {"Authorization": AUTH_TOKEN}
        with requests.get(download_url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f" -> Saved to {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {replay_id}. Error: {e}")


def main():
    """
    Main function to orchestrate fetching and downloading replays.
    """
    replays_to_download = fetch_replay_list(TARGET_RANK, REPLAY_COUNT)

    if not replays_to_download:
        print("Exiting.")
        return

    # Ensure the download directory exists
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    for replay in replays_to_download:
        print(replay["id"], replay["link"])
        print(replay["min_rank"])
        download_replay(replay, DOWNLOAD_DIR)
        time.sleep(1.5)

    print("\nDownload process finished.")


if __name__ == "__main__":
    main()
