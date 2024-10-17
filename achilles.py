import requests
import os

# Figshare shareable link (replace with your actual Figshare CSV link)
figshare_url = 'https://figshare.com/ndownloader/files/49843176'

def download_csv(save_directory):
    # Ensure the directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Full file path where the CSV will be saved
    output_file = os.path.join(save_directory, "achilles.csv")
    
    # Send a GET request to the Figshare link
    response = requests.get(figshare_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Write the content to a local CSV file
        with open(output_file, 'wb') as file:
            file.write(response.content)
        print(f"achilles.csv successfully downloaded and saved as {output_file}")
    else:
        print(f"Failed to download achilles.csv. HTTP Status code: {response.status_code}")