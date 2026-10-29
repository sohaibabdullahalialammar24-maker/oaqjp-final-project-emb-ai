"use strict";

window.RunSentimentAnalysis = () => {
    const textToAnalyze = document.getElementById("textToAnalyze").value;
    const responseContainer = document.getElementById("system_response");
    const request = new XMLHttpRequest();

    request.onreadystatechange = () => {
        if (request.readyState === XMLHttpRequest.DONE) {
            responseContainer.textContent = request.status >= 500
                ? "Emotion service is unavailable. Please try again later."
                : request.responseText;
            responseContainer.setAttribute(
                "role",
                request.status >= 400 ? "alert" : "status"
            );
        }
    };

    request.open(
        "GET",
        `/emotionDetector?textToAnalyze=${encodeURIComponent(textToAnalyze)}`,
        true
    );
    request.send();
};
