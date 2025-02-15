// Assuming you have an API endpoint that provides JSON data for requests
async function fetchRequests() {
    try {
      const response = await fetch('https://example.com/api/requests');
      if (!response.ok) throw new Error(`Request failed with status: ${response.status}`);

      const data = await response.json();

      // Convert JSON data to Request objects
      const requests = data.map((item) => ({
        ...item,  // Spread the object properties into a new object for TypeScript compatibility
        timestamp: new Date(item.timestamp),
        status: item.status === '1' ? 1 : 0
      }));

      return requests;
    } catch (error) {
      console.error('Error fetching data:', error);
      return [];
    }
  }