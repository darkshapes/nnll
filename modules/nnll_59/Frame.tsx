
// Image's metadata contain several regions
import imageUrl1 from "./assets/02-26-21_687138608095192_serenityFP8SD35LargeInclClips_v10.safetensors_00001_.png";
import imageUrl2 from "./assets/02-37-37_296936718115719_flux1-schnell.safetensors_00001_.png";
import ReactCompareImage from 'react-compare-image';
import "./index.css"

const windowWidth = window.innerWidth
const windowHeight = window.innerHeight

export function FrameDisplay() {
  // Will responsively zoom in on a meaningful
  // region:
  return (
    <>
    <div style={{width: windowWidth, height:windowHeight, background: "#0a0a0a"}}>
      <div style={{ width: windowWidth*.93, height:windowHeight, overflow: 'hidden'}}>
          <ReactCompareImage leftImage={imageUrl1} rightImage={imageUrl2} aspectRatio="wider"  />;

      </div>
    </div>
    </>
  );
}