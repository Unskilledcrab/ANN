using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NeuronController : MonoBehaviour
{
    public EventHandler MouseEnter;

    private void OnMouseEnter()
    {
        MouseEnter?.Invoke(this, EventArgs.Empty);
    }
}
