"use client";

import { useState, useEffect } from "react";
import { MdSupervisedUserCircle, MdCategory, MdPeopleAlt, MdCloudQueue } from "react-icons/md";
import styles from "./card.module.css";
import { getModelsCount, getUsersWithEmbeddingsCount, getUnknownPersonsCount } from "@/app/lib/actions";

const Card = ({ item }) => {
  const [count, setCount] = useState(item.number || 0);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const fetchDynamicData = async () => {
      try {
        setIsLoading(true);
        
        let result = 0;
        switch (item.type) {
          case "models":
            result = await getModelsCount();
            break;
          case "people":
            result = await getUsersWithEmbeddingsCount();
            break;
          case "unknown":
            result = await getUnknownPersonsCount();
            break;
          default:
            // For other types, use the provided number
            result = item.number || 0;
        }
        
        setCount(result);
      } catch (error) {
        console.error(`Error fetching ${item.type} count:`, error);
      } finally {
        setIsLoading(false);
      }
    };

    if (["models", "people", "unknown"].includes(item.type)) {
      fetchDynamicData();
    }
  }, [item.type, item.number]);

  // Select icon based on card type
  const getIcon = () => {
    switch (item.type) {
      case "models":
        return <MdCategory size={24} />;
      case "people":
        return <MdPeopleAlt size={24} />;
      case "latency":
        return <MdCloudQueue size={24} />;
      default:
        return <MdSupervisedUserCircle size={24} />;
    }
  };

  return (
    <div className={styles.container}>
      {getIcon()}
      <div className={styles.texts}>
        <span className={styles.title}>{item.title}</span>
        <span className={styles.number}>
          {isLoading ? "Loading..." : count}
          {item.unit && ` ${item.unit}`}
        </span>
        {item.subText && <span className={styles.subText}>{item.subText}</span>}
      </div>
    </div>
  );
};

export default Card;
