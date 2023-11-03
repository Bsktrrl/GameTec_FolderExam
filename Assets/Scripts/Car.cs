using System.Collections.Generic;
using UnityEngine;

public class Car : MonoBehaviour
{
    [Header("Car Stats")]
    public NeuralNetwork brain;

    public Vector3 carPosition = new Vector3();
    [SerializeField] Vector3 carVelocity = new Vector3();
    [SerializeField] Vector3 carAcceleration = new Vector3();
    [SerializeField] Vector3 carVelocity_Total = new Vector3();
    public float carVelocity_Max = 100;
    [SerializeField] float carMass = 10;

    public bool isAlive = true;
    public float timeAlive = 0f;
    public float timeWithoutCheckpoint = 0f;
    public int current_Checkpoint = 0;
    public int current_Lap = 0;

    [SerializeField] int steering_force = 50;
    [SerializeField] int forward_force = 50;

    [Header("Training")]
    public bool training = true;
    [SerializeField] float learningDeviation = 0.8f;


    //--------------------


    private void Awake()
    {
        carPosition = CarManager.instance.carSpawnPosition;
    }
    private void Start()
    {
        //Set the control panel of the car to match the desired data from previous generations
        SetupBrain();
    }
    private void Update()
    {
        //Check if the car should be despawned
        UpdateTimeAlive();


        Navigate();
        Movement();
    }


    //--------------------


    void SetupBrain()
    {
        if (brain == null)
        {
            brain = new NeuralNetwork();
            brain.Setup(5, new List<int>() { 32 }, 2);
        }
    }

    void UpdateTimeAlive()
    {
        //Despawn car if not alive
        if (!isAlive)
        {
            current_Checkpoint = 0;
            gameObject.SetActive(false);

            return;
        }

        timeAlive += Time.deltaTime;
        timeWithoutCheckpoint += Time.deltaTime;

        //Set the car to "dead" of it hasn't reach a new checkpoint before the time runs out
        if (timeWithoutCheckpoint > CarManager.instance.despawnTime)
        {
            timeWithoutCheckpoint = 0;
            isAlive = false;

            return;
        }
    }

    void Navigate()
    {
        //Use Raycasts to detect and navigate the area in front of the car
        List<float> raycastList = RaycastingFromCar();

        //Use the result from the raycast to insert information to the last half part of the neuron network
        List<float> outputList = brain.FeedForward(raycastList);

        //Perform "inputs" so that the car can move as if "WASD"-button was presses, based on the raycast information
        #region
        if (outputList[0] >= 0.5f)
        {
            CarForwardInput(Field(outputList[0], 0.5f, 1f, 0f, 1f));
        }
        else
        {
            CarBackwardsInput(Field(outputList[0], 0.5f, 0f, 0f, 1f));
        }

        if (outputList[1] >= 0.5f)
        {
            CarRightInput(Field(outputList[1], 0.5f, 1f, 0f, 1f));
        }
        else
        {
            CarLeftInput(Field(outputList[1], 0.5f, 0f, 0f, 1f));
        }
        #endregion

        if (training)
        {
            //Make a list<float> to set a new dircetion for the car
            List<float> direction = new List<float>();

            //Get the direction from all 5 angles attached to the car
            float forward = raycastList[0] / learningDeviation;
            float forward_Right = raycastList[1] / learningDeviation;
            float forward_Left = raycastList[2] / learningDeviation;
            float right = raycastList[3] / learningDeviation;
            float left = raycastList[4] / learningDeviation;

            //Calculate new floats in each direction to see which to give the most importance to
            float direction_Forward = (forward / 2f) + (forward_Right / 4f) + (forward_Left / 4);
            float direction_Back = 1 - direction_Forward;
            float direction_Right = (right * 0.5f) + (forward_Right * 0.5f);
            float direction_Left = (left * 0.5f) + (forward_Left * 0.5f);

            //Add the results into the "direction"-list
            direction.Add(direction_Forward);
            direction.Add(direction_Back);
            direction.Add(direction_Right);
            direction.Add(direction_Left);

            //Perform backpropegation through the Neural Network, so that the car can continue driving with numbers more accurate than earlier
            brain.BackPropagate(outputList, direction);
        }
    }
    List<float> RaycastingFromCar()
    {
        //Send out 5 Raycarsts in different direction from the car to detect walls in those directions
        List<float> inputList = new List<float>();
        int layerMask = 1 << 8;

        //In the front
        #region
        RaycastHit front_Hit;
        if (Physics.Raycast(transform.position, transform.TransformDirection(Vector3.forward), out front_Hit, 20, layerMask))
        {
            inputList.Add(Field(front_Hit.distance, 0, 20, -1f, 1f));
            Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.forward) * front_Hit.distance, Color.red);
        }
        else
        {
            inputList.Add(1f);
            Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.forward) * 20, Color.green);
        }
        #endregion

        //In the frontRight
        #region
        RaycastHit frontRight_Hit;
        if (Physics.Raycast(transform.position, transform.TransformDirection((Vector3.right + Vector3.forward).normalized), out frontRight_Hit, 20, layerMask))
        {
            inputList.Add(Field(frontRight_Hit.distance, 0, 20, -1f, 1f));
            Debug.DrawRay(transform.position, transform.TransformDirection((Vector3.right + Vector3.forward).normalized) * frontRight_Hit.distance, Color.red);
        }
        else
        {
            inputList.Add(1f);
            Debug.DrawRay(transform.position, transform.TransformDirection((Vector3.right + Vector3.forward).normalized) * 20, Color.green);
        }
        #endregion

        //In the frontLeft
        #region
        RaycastHit frontLeft_Hit;
        if (Physics.Raycast(transform.position, transform.TransformDirection((-Vector3.right + Vector3.forward).normalized), out frontLeft_Hit, 20, layerMask))
        {
            inputList.Add(Field(frontLeft_Hit.distance, 0, 20, -1f, 1f));
            Debug.DrawRay(transform.position, transform.TransformDirection((-Vector3.right + Vector3.forward).normalized) * frontLeft_Hit.distance, Color.red);
        }
        else
        {
            inputList.Add(1f);
            Debug.DrawRay(transform.position, transform.TransformDirection((-Vector3.right + Vector3.forward).normalized) * 20, Color.green);
        }
        #endregion

        //In the Right
        #region
        RaycastHit right_Hit;
        if (Physics.Raycast(transform.position, transform.TransformDirection(Vector3.right), out right_Hit, 20, layerMask))
        {
            inputList.Add(Field(right_Hit.distance, 0, 20, -1f, 1f));
            Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.right) * right_Hit.distance, Color.red);
        }
        else
        {
            inputList.Add(1f);
            Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.right) * 20, Color.green);
        }
        #endregion

        //In the Left
        #region
        RaycastHit left_Hit;
        if (Physics.Raycast(transform.position, transform.TransformDirection(-Vector3.right), out left_Hit, 20))
        {
            inputList.Add(Field(left_Hit.distance, 0, 20, -1f, 1f));
            Debug.DrawRay(transform.position, transform.TransformDirection(-Vector3.right) * left_Hit.distance, Color.red);
        }
        else
        {
            inputList.Add(1f);
            Debug.DrawRay(transform.position, transform.TransformDirection(-Vector3.right) * 20, Color.green);
        }
        #endregion

        return inputList;
    }


    //--------------------


    public static float Field(float value, float from1, float to1, float from2, float to2)
    {
        return (value - from1) / (to1 - from1) * (to2 - from2) + from2;
    }

    public void Movement()
    {
        //Perform the movement of the car based on its velocity and acceleration
        carVelocity += carAcceleration * Time.deltaTime;

        carPosition += carVelocity * Time.deltaTime;
        carVelocity_Total += carVelocity * Time.deltaTime;
        carAcceleration = Vector3.zero;

        transform.position = carPosition;

        if (carVelocity.magnitude > 1.5f)
        {
            transform.forward = carVelocity.normalized;
        }
    }
    public void AddForce(Vector3 force)
    {
        carAcceleration += force / carMass;
    }

    #region Car Inputs
    //For making the car move, inputs as with "WASD" must be added to the car's force and physics
    //The car uses a desired velocity to make a point it steers toward
    //It's important the car only steers a bit each fram and not al the way, to prevent oversteering
    public void CarForwardInput(float field)
    {
        Vector3 direction = carPosition;
        direction += transform.forward * field;

        Vector3 desired_velocity = (direction - carPosition).normalized * carVelocity_Max;
        Vector3 carSteering = desired_velocity - carVelocity;

        AddForce(carSteering * forward_force * field);
    }
    public void CarBackwardsInput(float field)
    {
        if (carVelocity.magnitude < 1)
        {
            return;
        }

        Vector3 direction = carPosition;
        direction -= transform.forward * field;

        Vector3 desired_velocity = (direction - carPosition).normalized * carVelocity_Max / 2;
        Vector3 carSteering = desired_velocity - carVelocity;

        AddForce(carSteering * forward_force * field);
    }
    public void CarRightInput(float field)
    {
        Vector3 direction = carPosition;
        direction += transform.right * carVelocity.sqrMagnitude * field;

        Vector3 desired_velocity = (direction - carPosition).normalized * carVelocity_Max / 2;
        Vector3 carSteering = desired_velocity - carVelocity;

        AddForce(carSteering * steering_force * field);
    }
    public void CarLeftInput(float field)
    {
        Vector3 direction = carPosition;
        direction -= transform.right * carVelocity.sqrMagnitude * field;

        Vector3 desired_velocity = (direction - carPosition).normalized * carVelocity_Max / 2;
        Vector3 carSteering = desired_velocity - carVelocity;

        AddForce(carSteering * steering_force * field);
    }
    #endregion


    //--------------------


    //Check different ways to despawn the car
    private void OnTriggerEnter(Collider other)
    {
        if (other.tag == "DeathBox")
        {
            isAlive = false;
            return;
        }
        
        if (other.tag != "Checkpoint")
        {
            isAlive = false;
            return;
        }

        if (other.tag == "Checkpoint")
        {
            timeWithoutCheckpoint = 0;
            ColliderIndex checkpoint = other.GetComponent<ColliderIndex>();

            //if moving into a previous checkpoint
            if (current_Checkpoint > checkpoint.index)
            {
                current_Checkpoint = 0;

                isAlive = false;
                return;
            }

            //If jumped over a checkpoint
            else if ((current_Checkpoint + 1) < checkpoint.index)
            {
                current_Checkpoint = 0;

                isAlive = false;
                return;
            }

            //If reaching the Goal
            else if (current_Checkpoint > 0 && checkpoint.index == 0)
            {
                current_Checkpoint = 0;

                isAlive = false;
                return;
            }

            //If moving backwards in the start
            else if (checkpoint.index == 0)
            {
                current_Checkpoint = 0;

                isAlive = false;
                return;
            }

            //If passing a checkpoint the inteded way, continue
            else
            {
                current_Checkpoint = checkpoint.index;
            }
        }
    }
}
