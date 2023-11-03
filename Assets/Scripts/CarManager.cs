using System;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

public class CarManager : MonoBehaviour
{
    public static CarManager instance; //Singleton

    [Header("Car Stats")]
    [SerializeField] GameObject car_Parent;
    [SerializeField] GameObject car_Prefab;
    [SerializeField] List<GameObject> carList = new List<GameObject>();

    [SerializeField] int carsToSpawn = 50;
    public int despawnTime = 10;
    public Vector3 carSpawnPosition = new Vector3(0, 0, -75);

    [Header("AI stats")]
    [SerializeField] int carsAlive;
    [SerializeField] float timeAlive;
    [SerializeField] float highestTimeAlive;
    [SerializeField] int generation = 1;
    [SerializeField] bool training = true;
    [SerializeField] int parentsAmount = 2;


    //--------------------


    private void Awake()
    {
        //Singleton
        if (instance != null && instance != this)
        {
            Destroy(gameObject);
        }
        else
        {
            instance = this;
        }
    }
    private void Start()
    {
        //Add x amount of card to the scene
        AddCar(carsToSpawn, carSpawnPosition);

        //Set "Cars Alive"-Parameter
        CheckCarsAlive();
    }
    private void Update()
    {
        //Update time cars have ben active
        timeAlive += Time.deltaTime;

        //Check if all cars have crashed
        CardCrashed();
    }


    //--------------------


    #region Add Cars
    void AddCar()
    {
        //Add a car and place it under its ObjectParent
        carList.Add(Instantiate(car_Prefab) as GameObject);
        carList[carList.Count - 1].transform.SetParent(car_Parent.transform);

        //Set if this car should be trained
        carList[carList.Count - 1].GetComponent<Car>().training = training;
    }
    void AddCar(Vector3 position)
    {
        //Add a car in a set position and place it under its ObjectParent
        carList.Add(Instantiate(car_Prefab) as GameObject);
        carList[carList.Count - 1].transform.SetParent(car_Parent.transform);
        carList[carList.Count - 1].transform.position.Set(position.x, position.y, position.z);

        //Set if this car should be trained
        carList[carList.Count - 1].GetComponent<Car>().training = training;
    }
    void AddCar(int amount, Vector3 position)
    {
        for (int i = 0; i < amount; i++)
        {
            //Add x amount of cars in a set position and place it under its ObjectParent
            carList.Add(Instantiate(car_Prefab) as GameObject);
            carList[carList.Count - 1].transform.SetParent(car_Parent.transform);
            carList[carList.Count - 1].transform.position.Set(position.x, position.y, position.z);

            //Set if the cars should be trained
            carList[carList.Count - 1].GetComponent<Car>().training = training;
        }
    }
    #endregion


    //--------------------


    void CardCrashed()
    {
        //Check if any car have crashed
        bool allCarsHasCrashed = true;
        for (int i = 0; i < carList.Count; i++)
        {
            Car car = carList[i].GetComponent<Car>();
            if (car.isAlive)
            {
                print("car.isAlive");
                allCarsHasCrashed = false;

                break;
            }
        }

        //Update "Cars Alive"-Parameter
        CheckCarsAlive();

        //If all cars have crashed, start new session with new Generation
        if (allCarsHasCrashed)
        {
            MakeNewGeneration();
            generation++;
        }
    }
    void CheckCarsAlive()
    {
        carsAlive = 0;

        //Build the "Cars Alive"-Parameter
        for (int i = 0; i < carList.Count; i++)
        {
            Car car = carList[i].GetComponent<Car>();
            if (car.isAlive)
            {
                carsAlive++;
            }
        }
    }
    void MakeNewGeneration()
    {
        //Set Highest Time to have a number to compare with
        highestTimeAlive = timeAlive;
        timeAlive = 0;

        //Build a new list containing the best performing cards, based on time alive in the generation
        List<GameObject> newParents = new List<GameObject>();
        for (int n = 0; n < parentsAmount; n++)
        {
            float bestTimeLastingCar = 0;
            int parentIndex = 0;

            //Get the car with the best time
            for (int i = 0; i < carList.Count; i++)
            {
                if (carList[i].GetComponent<Car>().timeAlive > bestTimeLastingCar)
                {
                    bestTimeLastingCar = carList[i].GetComponent<Car>().timeAlive;
                    parentIndex = i;
                }
            }

            //Add this car to the list
            newParents.Add(carList[parentIndex]);
        }

        //Destroy the rest of the cars
        for (int i = carList.Count - 1; i >= 0; i--)
        {
            Destroy(carList[i]);
            carList.RemoveAt(i);
        }

        //Mutate the best cars into a new generation
        int newParent_index = 0;

        for (int i = parentsAmount; i < carsToSpawn; i++)
        {
            AddCar(carSpawnPosition);

            Car child = carList[carList.Count - 1].GetComponent<Car>();
            child.training = training;

            Car parent = newParents[newParent_index].GetComponent<Car>();
            child.brain = parent.brain.Copy();
            child.brain.GeneticAlgorithm();
            newParent_index = (newParent_index + 1) % newParents.Count;
        }

        for (int i = newParents.Count - 1; i >= 0; i--)
        {
            AddCar(carSpawnPosition);

            Car child = carList[carList.Count - 1].GetComponent<Car>();
            child.training = training;

            child.brain = newParents[i].GetComponent<Car>().brain.Copy();
            if (i == 0)
            {
                child.isReadyForDebug = true;
            }

            GameObject parentToDestroy = newParents[i];
            newParents.RemoveAt(i);
            Destroy(parentToDestroy);
        }

        //Set new spawn position of the new generation
        //for (int i = 0; i < carsToSpawn; i++)
        //{
        //    carList[i].transform.position = carSpawnPosition;
        //    Car car = carList[i].GetComponent<Car>();
        //    car.carPosition = carSpawnPosition;
        //    car.isAlive = true;
        //}
    }
}
