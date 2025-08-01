"use client";

const Error = ({ error }) => {
  return (
    <div>
      <h1>An Error Occurred</h1>
      <p>{error?.message || "Unknown error"}</p>
      <pre>{error?.stack || "No stack trace available"}</pre>
    </div>
  );
};

export default Error;