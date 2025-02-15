import * as React from "react";
import { render } from "react-dom";
import Reorder, {
  reorder,
  reorderImmutable,
  reorderFromTo,
  reorderFromToImmutable
} from "react-reorder";
import move from "lodash-move";

import "./interact.css";

export default function App() {
  const [list, setList] = React.useState([
    "qeqwe",
    "jjshh",
    "piaif",
    "asjdhgj"
  ]);

  const onReorder = (e, from, to) => {
    setList(move(list, from, to));
  };

  return (
    <div className="App">
      <h1>React Reorder</h1>
      <hr />
      <div
        style={{
          background: "#ddd",
          padding: 20
        }}
      >
        <Reorder
          reorderId="my-list" // Unique ID that is used internally to track this list (required)
          reorderGroup="reorder-group" // A group ID that allows items to be dragged between lists of the same group (optional)
          // getRef={this.storeRef.bind(this)} // Function that is passed a reference to the root node when mounted (optional)
          component="div" // Tag name or Component to be used for the wrapping element (optional), defaults to 'div'
          placeholderClassName="placeholder" // Class name to be applied to placeholder elements (optional), defaults to 'placeholder'
          draggedClassName="dragged" // Class name to be applied to dragged elements (optional), defaults to 'dragged'
          lock="horizontal" // Lock the dragging direction (optional): vertical, horizontal (do not use with groups)
          holdTime={500} // Default hold time before dragging begins (mouse & touch) (optional), defaults to 0
          touchHoldTime={500} // Hold time before dragging begins on touch devices (optional), defaults to holdTime
          mouseHoldTime={200} // Hold time before dragging begins with mouse (optional), defaults to holdTime
          onReorder={onReorder} // Callback when an item is dropped (you will need this to update your state)
          autoScroll={true} // Enable auto-scrolling when the pointer is close to the edge of the Reorder component (optional), defaults to true
          disabled={false} // Disable reordering (optional), defaults to false
          disableContextMenus={true} // Disable context menus when holding on touch devices (optional), defaults to true
          placeholder={
            <div className="custom-placeholder" /> // Custom placeholder element (optional), defaults to clone of dragged element
          }
        >
          {list.map(item => (
            <div
              style={{
                height: 50,
                background: "grey",
                margin: "10px 0px",
                cursor: "pointer"
              }}
              key={item}
            >
              {item}
            </div>
          ))

          }
        </Reorder>
      </div>
    </div>
  );
}
