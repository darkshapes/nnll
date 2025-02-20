import * as React from "react"
import "./bg.css"
import { useKeenSlider } from "keen-slider/react"
import "keen-slider/keen-slider.min.css"

// const FullScreenSlider = (slider) => {
//   slider.size(100,100)
//     })
//   })

export default function BackgroundApp() {
  const [rotation, setRotation] = React.useState({})
  const [lastProgress, setLastProgress] = React.useState(0)

  const [sliderRef] = useKeenSlider<HTMLDivElement>({
    slides: 2,
    detailsChanged(s) {
      const progress = s.track.details.progress
      const delta = lastProgress - progress
      setLastProgress(progress)
      setRotation(delta * 360)
    },
    loop: true,
  })

  return (
    <div
      style={{
        backgroundImage: `linear-gradient(${rotation}deg, black 0px, black 50%, white 50%, white 100%)`,
      }}
      className="background-rotation"
      ref={sliderRef}
    >
      <div
        className="background-rotation__inner"
        style={{
          backgroundImage: `linear-gradient(${rotation}deg, white 0px, white 50%, black 50%, black 100%)`,
        }}
      >
        <span>keen-slider</span>
      </div>
    </div>
  )
}
