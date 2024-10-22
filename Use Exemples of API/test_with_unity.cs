using System.Collections;
using UnityEngine;
using UnityEngine.Networking;


// definition of the request structure
[System.Serializable]
public class ModelRequest
{
    public string prompt; // Prompt Attribut 
}

// Definition of the response structure
[System.Serializable]
public class ModelResponse
{
    public string model_3d_path; 
    public string download_url;  
}

public class AssetGeneration : MonoBehaviour
{
    // URL 
    private string apiUrl = "http://11.0.31.197:8000/generate3dmodel"; // use your own API URL here

    void Start()
    {
    
        // Lunch the request when the game starts.
        Request3DModel("a kangoroo");
    }

    // Model generation method. Pass the prompt as an argument.
    public void Request3DModel(string prompt)
    {
        StartCoroutine(PostRequest(prompt));
    }

    // Coroutine for sending the POST request 
    IEnumerator PostRequest(string prompt)
    {
        ModelRequest modelRequest = new ModelRequest { prompt = prompt };
        // Convert into JSON
        string jsonData = JsonUtility.ToJson(modelRequest); 
        Debug.Log("Sending JSON: " + jsonData);

        // Configuration of the POST request
        using (UnityWebRequest request = new UnityWebRequest(apiUrl, "POST"))
        {
            byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonData);
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            // sending the request and wait for the response
            yield return request.SendWebRequest();

            Debug.Log("Status Code: " + request.responseCode);

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Error: " + request.error);
                Debug.LogError("Response: " + request.downloadHandler.text);
            }
            else
            {
                Debug.Log("Response received: " + request.downloadHandler.text);
                ModelResponse response = JsonUtility.FromJson<ModelResponse>(request.downloadHandler.text);
                // Lunch the model downloading
                StartCoroutine(DownloadModel(response.download_url)); 
            }
        }
    }

    // Coroutine for the model downloading
    IEnumerator DownloadModel(string url)
    {
        using (UnityWebRequest request = UnityWebRequest.Get(url))
        {
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.ConnectionError || request.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.LogError("Download error: " + request.error);
            }
            else
            {
                Debug.Log("Model downloaded successfully.");

                // define the path for the downloaded model
                string filePath = Application.dataPath + "/Models/downloaded_model.obj";

                // Create the directory if it doesn't exist
                System.IO.Directory.CreateDirectory(Application.dataPath + "/Models");

                // write the model to disk
                System.IO.File.WriteAllBytes(filePath, request.downloadHandler.data);

                Debug.Log("Model saved to: " + filePath);
            }
        }
    }

}
