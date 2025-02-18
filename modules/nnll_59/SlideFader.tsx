import * as React from "react"
import "./styles.css"
import { useKeenSlider } from "keen-slider/react"
import "keen-slider/keen-slider.min.css"
import imageUrl1 from "./assets/02-26-21_687138608095192_serenityFP8SD35LargeInclClips_v10.safetensors_00001_.png";
import imageUrl2 from "./assets/02-37-37_296936718115719_flux1-schnell.safetensors_00001_.png";
import ReactCompareImage from 'react-compare-image';
import "./index.css"

const images = [
  imageUrl1,
  imageUrl2,
]
const windowWidth = window.innerWidth
const windowHeight = window.innerHeight

export default function SlideFader() {
  const [opacities, setOpacities] = React.useState<number[]>([])

  const [sliderRef] = useKeenSlider<HTMLDivElement>({
    slides: images.length,
    loop: true,
    detailsChanged(s) {
      const new_opacities = s.track.details.slides.map((slide) => slide.portion)
      setOpacities(new_opacities)
    },
  })

  return (
    <div ref={sliderRef} className="fader">
      {images.map((src, idx) => (
        <div
          key={idx}
          className="fader__slide"
          style={{ opacity: opacities[idx] }}
        >
        <ReactCompareImage leftImage={src} rightImage={src+1} aspectRatio="wider"  />;
        </div>
      ))}
    </div>
  )
}
