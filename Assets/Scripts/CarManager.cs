using System.Collections.Generic;
using UnityEngine;

public class CarManager : MonoBehaviour
{
    public static CarManager instance; //Singleton

    #region Variables
    [Header("Car Stats")]
    [SerializeField] GameObject car_Parent;
    [SerializeField] GameObject car_Prefab;
    [SerializeField] List<GameObject> carList = new List<GameObject>();

    [SerializeField] int carsToSpawn = 50;
    public int despawnTime = 10;
    public Vector3 carSpawnPosition = new Vector3(0, 0, -75);

    [Header("AI stats")]
    public int carsAlive;
    public float timeAlive;
    public float highestTimeAlive;
    public float bestTimeTotal = 0;
    public int generation = 1;
    [SerializeField] bool training = true;
    [SerializeField] int parentsAmount = 2;
    public bool endOfgeneration;
    #endregion


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

            if (highestTimeAlive > 2)
            {
                generation++;
                endOfgeneration = true;
            }
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

        if (highestTimeAlive > bestTimeTotal)
        {
            bestTimeTotal = highestTimeAlive;
        }

        timeAlive = 0;

        //Build a new list containing the best performing cards, based on time alive in the generation
        List<GameObject> parent = new List<GameObject>();
        int parentIndex = 0;
        for (int n = 0; n < parentsAmount; n++)
        {
            float bestTimeLastingCar = 0;
            parentIndex = 0;

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
            parent.Add(carList[parentIndex]);
        }

        //Destroy the all cars from the original list
        for (int i = carList.Count - 1; i >= 0; i--)
        {
            Destroy(carList[i]);
            carList.RemoveAt(i);
        }

        parentIndex = 0;

        //Mutate the best cars to make better prerequisites for the next generation of cars
        for (int i = parentsAmount; i < carsToSpawn; i++)
        {
            //Add the remaining cars to reach the desired spawned cars
            AddCar(carSpawnPosition);

            //Set the car to be able to train
            Car child = carList[carList.Count - 1].GetComponent<Car>();
            child.training = training;

            //Make a parent, which the child-car will look up to and mimic, to make higher chances of lasting longer
            Car _parent = parent[parentIndex].GetComponent<Car>();
            child.brain = _parent.brain.Copy();
            child.brain.Mutate();
            parentIndex = (parentIndex + 1) % parent.Count;
        }

        //Repeat the process, this time for the spots of the parent
        for (int i = parent.Count - 1; i >= 0; i--)
        {
            AddCar(carSpawnPosition);

            Car child = carList[carList.Count - 1].GetComponent<Car>();
            child.training = training;

            child.brain = parent[i].GetComponent<Car>().brain.Copy();

            //Destroy the parent
            Destroy(parent[i]);
            parent.RemoveAt(i);
        }
    }
}
