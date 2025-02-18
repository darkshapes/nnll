// SPDX-License-Identifier: blessing
// d a r k s h a p e s


import { useState } from "react"
import "./styles.css"
import { useKeenSlider, KeenSliderPlugin, KeenSliderInstance } from "keen-slider/react"
// import "keen-slider/keen-slider.min.css"

const MutationPlugin = (slider: KeenSliderInstance) => {
  const observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (_mutation) {
      slider.update()
    })
  })
  const config = { childList: true }

  slider.on("created", () => {
    observer.observe(slider.container, config)
  })
  slider.on("destroyed", () => {
    observer.disconnect()
  })
}


const WheelControls: KeenSliderPlugin = (slider) => {
  let touchTimeout: ReturnType<typeof setTimeout>
  let position: {
    x: number
    y: number
  }
  let wheelActive: boolean

  function dispatch(e: WheelEvent, name: string) {
    position.x -= e.deltaX
    position.y -= e.deltaY
    slider.container.dispatchEvent(
      new CustomEvent(name, {
        detail: {
          x: position.x,
          y: position.y,
        },
      })
    )
  }

  function wheelStart(e: WheelEvent) {
    position = {
      x: e.pageX,
      y: e.pageY,
    }
    dispatch(e, "ksDragStart")
  }

  function wheel(e: WheelEvent) {
    dispatch(e, "ksDrag")
  }

  function wheelEnd(e: WheelEvent) {
    dispatch(e, "ksDragEnd")
  }

  function eventWheel(e: WheelEvent) {
    e.preventDefault()
    if (!wheelActive) {
      wheelStart(e)
      wheelActive = true
    }
    wheel(e)
    clearTimeout(touchTimeout)
    touchTimeout = setTimeout(() => {
      wheelActive = false
      wheelEnd(e)
    }, 50)
  }

  slider.on("created", () => {
    slider.container.addEventListener("wheel", eventWheel, {
      passive: false,
    })
  })
}

export default function App() {
  const [slides, setSlides] = useState([1])
  const [sliderRef] = useKeenSlider<HTMLDivElement>(
    {
      mode: "free-snap",
      loop: false,
      rubberband: true,
      vertical: true,
      slides: {
        origin: "center",
        perView: 7,
        spacing: 15,
      },
    },
    [WheelControls, MutationPlugin]
  )

  return (
    <>
    <div ref={sliderRef} className="keen-slider" style={{ width: 100 }}>
        {slides.map((slide) => {
          return (
            <div
              key={slide}
              className={"keen-slider__slide number-slide" + ((slide % 5) + 1)}
              onClick={() => setSlides([...slides, slides.length + 1])}
              onContextMenu={() => setSlides(slides.slice(0, -1))}
            >
              {slide}
            </div>
          )
        })}
      </div>

      </>
  )
}
