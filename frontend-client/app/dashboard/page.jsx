import { fetchCardData } from "@/app/lib/data";
import Card from "@/app/ui/dashboard/card/card";
import styles from "../ui/dashboard/dashboard.module.css";
import Rightbar from "../ui/dashboard/rightbar/rightbar";
import Detections from "@/app/ui/dashboard/transactions/transactions";
import dynamic from "next/dynamic";

// Import chart with dynamic loading to avoid SSR issues with recharts
const ChartWithNoSSR = dynamic(
  () => import("../ui/dashboard/chart/chart"),
  { ssr: false }
);

const Dashboard = async () => {
  const cardsData = await fetchCardData();

  return (
    <div className={styles.wrapper}>
      <div className={styles.main}>
        <div className={styles.cards}>
          {cardsData.map((item) => (
            <Card key={item.id} item={item} />
          ))}
        </div>
        <Detections />
        <div className={styles.chartSection}>
          <ChartWithNoSSR />
        </div>
      </div>
      <div className={styles.side}>
        <Rightbar />
      </div>
    </div>
  );
};

export default Dashboard;