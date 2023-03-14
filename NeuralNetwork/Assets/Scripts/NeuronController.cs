using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

public class NeuronController : MonoBehaviour, IPointerEnterHandler
{
    public EventHandler MouseEnter;

    public void OnPointerEnter(PointerEventData eventData)
    {
        MouseEnter?.Invoke(this, EventArgs.Empty);
    }

    private void OnMouseEnter()
    {
        MouseEnter?.Invoke(this, EventArgs.Empty);
    }
}
