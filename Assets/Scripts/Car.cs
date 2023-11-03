using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UIElements;
using UnityEngine.Windows;

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

    //[HideInInspector] public float despawnTime;
    public bool isAlive = true;
    public float timeAlive = 0f;
    public float timeWithoutCheckpoint = 0f;
    public int current_Checkpoint = 0;
    public int current_Lap = 0;

    [SerializeField] int steering_force = 50;
    [SerializeField] int forward_force = 50;

    [Header("Training")]
    public bool training = true;
    [SerializeField] float learning_Deviation = 0.8f;
    public bool isReadyForDebug = false;


    //--------------------


    private void Start()
    {
        carPosition = CarManager.instance.carSpawnPosition;

        SetupBrain();
    }
    private void Update()
    {
        UpdateTimeAlive();

        Training();
        Move();
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
        //Stop car if it's not alive
        if (!isAlive)
        {
            current_Checkpoint = 0;
            gameObject.SetActive(false);

            return;
        }

        timeAlive += Time.deltaTime;
        timeWithoutCheckpoint += Time.deltaTime;

        if (timeWithoutCheckpoint > CarManager.instance.despawnTime)
        {
            print("Car is Dead");

            timeWithoutCheckpoint = 0;

            isAlive = false;
            return;
        }
    }

    void Training()
    {
        List<float> raycastList = Raycast();
        List<float> outputList = brain.FeedForward(raycastList);

        if (isReadyForDebug)
        {
            //Debug.Log(outputList[0] + "," + outputList[1]);
        }

        if (outputList[0] >= 0.5f)
        {
            AddForwardInput(Map(outputList[0], 0.5f, 1f, 0f, 1f));
        }
        else
        {
            AddBackwardsInput(Map(outputList[0], 0.5f, 0f, 0f, 1f));
        }

        if (outputList[1] >= 0.5f)
        {
            AddRightInput(Map(outputList[1], 0.5f, 1f, 0f, 1f));
        }
        else
        {
            AddLeftInput(Map(outputList[1], 0.5f, 0f, 0f, 1f));
        }

        if (training)
        {
            List<float> targets = new List<float>();

            float forward_safety = raycastList[0] / learning_Deviation;
            float forward_right_safety = raycastList[1] / learning_Deviation;
            float forward_left_safety = raycastList[2] / learning_Deviation;
            float right_safety = raycastList[3] / learning_Deviation;
            float left_safety = raycastList[4] / learning_Deviation;

            float forward_target = (forward_safety / 2f) + (forward_right_safety / 4f) + (forward_left_safety / 4);
            float back_target = 1 - forward_target;
            float right_target = (right_safety * 0.5f) + (forward_right_safety * 0.5f);
            float left_target = (left_safety * 0.5f) + (forward_left_safety * 0.5f);

            targets.Add(forward_target);
            targets.Add(back_target);
            targets.Add(right_target);
            targets.Add(left_target);

            brain.BackPropagate(outputList, targets);
        }
    }
    List<float> Raycast()
    {
        List<float> inputList = new List<float>();
        int layerMask = 1 << 8;

        #region Different Raycast Hits
        //Hit from front of the Car
        RaycastHit front_Hit;
        if (Physics.Raycast(transform.position, transform.TransformDirection(Vector3.forward), out front_Hit, 20, layerMask))
        {
            inputList.Add(Map(front_Hit.distance, 0, 20, -1f, 1f));
            Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.forward) * front_Hit.distance, Color.red);
        }
        else
        {
            inputList.Add(1f);
            Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.forward) * 20, Color.green);
        }

        //Hit from frontRight of the Car
        RaycastHit frontRight_Hit;
        if (Physics.Raycast(transform.position, transform.TransformDirection((Vector3.right + Vector3.forward).normalized), out frontRight_Hit, 20, layerMask))
        {
            inputList.Add(Map(frontRight_Hit.distance, 0, 20, -1f, 1f));
            Debug.DrawRay(transform.position, transform.TransformDirection((Vector3.right + Vector3.forward).normalized) * frontRight_Hit.distance, Color.red);
        }
        else
        {
            inputList.Add(1f);
            Debug.DrawRay(transform.position, transform.TransformDirection((Vector3.right + Vector3.forward).normalized) * 20, Color.green);
        }

        //Hit from frontLeft of the Car
        RaycastHit frontLeft_Hit;
        if (Physics.Raycast(transform.position, transform.TransformDirection((-Vector3.right + Vector3.forward).normalized), out frontLeft_Hit, 20, layerMask))
        {
            inputList.Add(Map(frontLeft_Hit.distance, 0, 20, -1f, 1f));
            Debug.DrawRay(transform.position, transform.TransformDirection((-Vector3.right + Vector3.forward).normalized) * frontLeft_Hit.distance, Color.red);
        }
        else
        {
            inputList.Add(1f);
            Debug.DrawRay(transform.position, transform.TransformDirection((-Vector3.right + Vector3.forward).normalized) * 20, Color.green);
        }

        //Hit from Right of the Car
        RaycastHit right_Hit;
        if (Physics.Raycast(transform.position, transform.TransformDirection(Vector3.right), out right_Hit, 20, layerMask))
        {
            inputList.Add(Map(right_Hit.distance, 0, 20, -1f, 1f));
            Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.right) * right_Hit.distance, Color.red);
        }
        else
        {
            inputList.Add(1f);
            Debug.DrawRay(transform.position, transform.TransformDirection(Vector3.right) * 20, Color.green);
        }

        //Hit from Left of the Car
        RaycastHit left_Hit;
        if (Physics.Raycast(transform.position, transform.TransformDirection(-Vector3.right), out left_Hit, 20))
        {
            inputList.Add(Map(left_Hit.distance, 0, 20, -1f, 1f));
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


    public static float Map(float value, float from1, float to1, float from2, float to2)
    {
        return (value - from1) / (to1 - from1) * (to2 - from2) + from2;
    }

    public void Move()
    {
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
    public void AddForwardInput(float input)
    {
        Vector3 target = carPosition;
        target += transform.forward * input;
        Vector3 desired_vel = (target - carPosition).normalized * carVelocity_Max;
        Vector3 steering = desired_vel - carVelocity;
        AddForce(steering * forward_force * input);
    }
    public void AddBackwardsInput(float input)
    {
        if (carVelocity.magnitude < 1)
        {
            return;
        }

        Vector3 target = carPosition;
        target -= transform.forward * input;
        Vector3 desired_vel = (target - carPosition).normalized * carVelocity_Max / 2;
        Vector3 steering = desired_vel - carVelocity;
        AddForce(steering * forward_force * input);
    }
    public void AddRightInput(float input)
    {
        Vector3 target = carPosition;
        target += transform.right * carVelocity.sqrMagnitude * input;

        Vector3 desired_vel = (target - carPosition).normalized * carVelocity_Max / 2;
        Vector3 steering = desired_vel - carVelocity;
        AddForce(steering * steering_force * input);
    }
    public void AddLeftInput(float input)
    {
        Vector3 target = carPosition;
        target -= transform.right * carVelocity.sqrMagnitude * input;

        Vector3 desired_vel = (target - carPosition).normalized * carVelocity_Max / 2;
        Vector3 steering = desired_vel - carVelocity;
        AddForce(steering * steering_force * input);
    }
    #endregion


    //--------------------


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
            else
            {
                current_Checkpoint = checkpoint.index;
            }
        }
    }
}
