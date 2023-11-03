using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class UIManager : MonoBehaviour
{
    [SerializeField] TextMeshProUGUI carsAlive;
    [SerializeField] TextMeshProUGUI timeUsed;
    [SerializeField] TextMeshProUGUI generation;
    [SerializeField] TextMeshProUGUI bestTime;

    [SerializeField] TextMeshProUGUI HighScore;
    List<float> highScoreList = new List<float>();


    //--------------------


    private void Update()
    {
        //Set Cars Alive
        carsAlive.text = "Cars Alive: " + CarManager.instance.carsAlive;

        //Set Timer
        int minutes = Mathf.FloorToInt(CarManager.instance.timeAlive / 60);
        int seconds = Mathf.FloorToInt(CarManager.instance.timeAlive % 60);
        int milliseconds = Mathf.FloorToInt((CarManager.instance.timeAlive * 1000) % 1000);

        timeUsed.text = "Time: " + string.Format("{0:D2}:{1:D2}:{2:D3}", minutes, seconds, milliseconds);

        //Set Generation
        generation.text = "Generation: " + CarManager.instance.generation;

        //Set Best Time
        minutes = Mathf.FloorToInt(CarManager.instance.bestTimeTotal / 60);
        seconds = Mathf.FloorToInt(CarManager.instance.bestTimeTotal % 60);
        milliseconds = Mathf.FloorToInt((CarManager.instance.bestTimeTotal * 1000) % 1000);

        bestTime.text = "Best Time:\n" + string.Format("{0:D2}:{1:D2}:{2:D3}", minutes, seconds, milliseconds);

        //Set HighScoreList
        if (CarManager.instance.endOfgeneration)
        {
            if (highScoreList.Count <= 0)
            {
                highScoreList.Add(CarManager.instance.bestTimeTotal);
            }
            else if (CarManager.instance.bestTimeTotal > highScoreList[0])
            {
                highScoreList.Insert(0, CarManager.instance.bestTimeTotal);
            }

            if (highScoreList.Count > 16)
            {
                highScoreList.RemoveAt(highScoreList.Count - 1);
            }

            HighScore.text = "HighScore Table:\n";
            for (int i = 0; i < highScoreList.Count; i++)
            {
                minutes = Mathf.FloorToInt(highScoreList[i] / 60);
                seconds = Mathf.FloorToInt(highScoreList[i] % 60);
                milliseconds = Mathf.FloorToInt((highScoreList[i] * 1000) % 1000);

                HighScore.text += string.Format("{0:D2}:{1:D2}:{2:D3}", minutes, seconds, milliseconds) + "\n";
            }

            CarManager.instance.endOfgeneration = false;
        }
    }
}
