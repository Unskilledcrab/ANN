using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    public float moveSpeed = 5.0f; // The player's movement speed
    public float turnSpeed = 90; // degrees per second
    public float jumpSpeed = 8;
    
    private float gravity = 9.8f;
    private float currentVSpeed = 0;
    private CharacterController controller;

    private void Start()
    {
        // Get a reference to the Character Controller component attached to the player
        controller = GetComponent<CharacterController>();
    }

    private void Update()
    {
        // Get the player's input for movement
        float moveHorizontal = Input.GetAxis("Horizontal");
        float moveVertical = Input.GetAxis("Vertical");

        transform.Rotate(0, Input.GetAxis("Mouse X") * turnSpeed * Time.deltaTime, 0);
        if (controller.isGrounded)
        {
            currentVSpeed = 0; // grounded character has vSpeed = 0...
            if (Input.GetKeyDown("space"))
            { // unless it jumps:
                currentVSpeed = jumpSpeed;
            }
        }

        // Create a movement vector based on the input and the player's speed
        Vector3 movement = new Vector3(moveHorizontal, 0.0f, moveVertical) * moveSpeed;

        // apply gravity acceleration to vertical speed:
        currentVSpeed -= gravity * Time.deltaTime;
        movement.y = currentVSpeed; // include vertical speed in vel
        
        // convert vel to displacement and Move the character:
        controller.Move(movement * Time.deltaTime);
    }
}
