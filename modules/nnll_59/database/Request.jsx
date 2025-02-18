import {useState, useEffect, fetchRequests} from 'react';

function App() {
  const [requests, setRequests] = useState([]);

  useEffect(() => {
    fetchRequests().then((newRequests) => {
      setRequests(newRequests);
    });
  }, []);

  return (
    // Render the list of requests with their details
    <ul>
      {requests.map((request) => (
        <li key={request.id}>
          <h2>{request.prompt}</h2>
          <p><strong>Status:</strong> {request.status}</p>
          {/* Display other relevant information */}
        </li>
      ))}
    </ul>
  );
}