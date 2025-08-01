"use client";

import styles from "@/app/ui/dashboard/help/help.module.css";
import { MdExpandMore, MdSearch, MdOutlineCode, MdCloud, MdDevices, MdFace, MdVisibility } from "react-icons/md";
import { useState } from "react";
import Image from "next/image";

const HelpPage = () => {
  const [activeTab, setActiveTab] = useState("help");
  const [expandedFaq, setExpandedFaq] = useState(null);
  
  const toggleFaq = (index) => {
    if (expandedFaq === index) {
      setExpandedFaq(null);
    } else {
      setExpandedFaq(index);
    }
  };
  
  const faqs = [
    {
      question: "How do I add a new model?",
      answer: "You can add a new model by navigating to the Models page and clicking on the 'Add New' button. Fill in all the required fields including model title, category, accuracy metrics, and edge device compatibility. Make sure to provide a detailed description of your model for better documentation."
    },
    {
      question: "How do I connect an edge device?",
      answer: "To connect an edge device, go to the Edge Devices page and click 'Add New Device'. Follow the setup wizard to configure your device and establish the connection. You'll need to specify the device type (Arduino Nicla or Raspberry Pi), set up network parameters, and deploy the corresponding model to the device."
    },
    {
      question: "How can I view real-time analytics?",
      answer: "Real-time analytics can be viewed on the Real-time Data page. You'll see live updates from all connected devices and can filter data based on various parameters including device type, accuracy thresholds, and detection confidence scores. The dashboard provides visualization tools to better understand the data patterns."
    },
    {
      question: "How do I set up user permissions?",
      answer: "User permissions can be managed in the Users section. Select a user and click on 'Edit Permissions' to assign different access levels and capabilities. Admin users can manage all aspects of the system, while regular users may have limited access to certain features based on their assigned roles."
    },
    {
      question: "What is the difference between model categories?",
      answer: "The system supports four model categories: Base (original full-size models), Distilled (knowledge-distilled smaller models), Quantized (models with reduced numerical precision), and Distilled-Quantized (models that have undergone both distillation and quantization for maximum efficiency on edge devices)."
    }
  ];

  const teamMembers = [
    {
      name: "Brandon",
      role: "Backend Developer",
      image: "/avatar-4.png",
      description: "Responsible for server architecture, database management, and API development. Ensures smooth data flow between edge devices and cloud services."
    },
    {
      name: "Yousuf",
      role: "AI Engineer",
      image: "/avatar-1.png",
      description: "Specializes in machine learning model development, optimization, and training. Focuses on improving facial recognition accuracy in challenging conditions."
    },
    {
      name: "Edison",
      role: "AI Engineer",
      image: "/avatar-2.png",
      description: "Works on TinyML model compression, quantization, and knowledge distillation. Ensures models can run efficiently on resource-constrained edge devices."
    },
    {
      name: "Shadman",
      role: "Frontend Developer",
      image: "/avatar-3.png",
      description: "Creates intuitive user interfaces and dashboards. Ensures seamless user experience across the platform with responsive design."
    }
  ];

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h1 className={styles.title}>
          {activeTab === "help" && "Help Center"}
          {activeTab === "team" && "Our Team"}
          {activeTab === "project" && "About the Project"}
        </h1>
        <p className={styles.subtitle}>
          {activeTab === "help" && "Find answers to common questions and learn how to use our platform"}
          {activeTab === "team" && "Meet the talented individuals behind this project"}
          {activeTab === "project" && "Discover our innovative hybrid edge-cloud architecture"}
        </p>
      </div>

      <div className={styles.tabs}>
        <button 
          className={`${styles.tab} ${activeTab === "help" ? styles.activeTab : ""}`}
          onClick={() => setActiveTab("help")}
        >
          Help Center
        </button>
        <button 
          className={`${styles.tab} ${activeTab === "team" ? styles.activeTab : ""}`}
          onClick={() => setActiveTab("team")}
        >
          Our Team
        </button>
        <button 
          className={`${styles.tab} ${activeTab === "project" ? styles.activeTab : ""}`}
          onClick={() => setActiveTab("project")}
        >
          About the Project
        </button>
      </div>

      {activeTab === "help" && (
        <>
          <div className={styles.searchContainer}>
            <div className={styles.searchBox}>
              <MdSearch className={styles.searchIcon} />
              <input type="text" placeholder="Search for help topics..." />
              <button className={styles.searchButton}>Search</button>
            </div>
          </div>

          <div className={styles.categories}>
            <div className={styles.category}>Getting Started</div>
            <div className={styles.category}>Models</div>
            <div className={styles.category}>Edge Devices</div>
            <div className={styles.category}>Analytics</div>
            <div className={styles.category}>Account</div>
          </div>

          <div className={styles.faqSection}>
            <h2>Frequently Asked Questions</h2>
            
            <div className={styles.faqs}>
              {faqs.map((faq, index) => (
                <div key={index} className={styles.faqItem}>
                  <div 
                    className={styles.faqQuestion} 
                    onClick={() => toggleFaq(index)}
                  >
                    <h3>{faq.question}</h3>
                    <MdExpandMore className={`${styles.expandIcon} ${expandedFaq === index ? styles.expanded : ''}`} />
                  </div>
                  <div className={`${styles.faqAnswer} ${expandedFaq === index ? styles.expanded : ''}`}>
                    {faq.answer}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className={styles.supportSection}>
            <h2>Need more help?</h2>
            <p>Our support team is available 24/7 to assist you with any questions or issues.</p>
            <div className={styles.supportOptions}>
              <button className={styles.supportButton}>Contact Support</button>
              <button className={styles.docsButton}>Documentation</button>
            </div>
          </div>
        </>
      )}

      {activeTab === "team" && (
        <div className={styles.teamSection}>
          <div className={styles.teamGrid}>
            {teamMembers.map((member, index) => (
              <div key={index} className={styles.teamCard}>
                <div className={styles.memberImageContainer}>
                  <Image 
                    src={member.image} 
                    alt={member.name} 
                    width={120} 
                    height={120} 
                    className={styles.memberImage} 
                  />
                </div>
                <h3 className={styles.memberName}>{member.name}</h3>
                <div className={styles.memberRole}>{member.role}</div>
                <p className={styles.memberDescription}>{member.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === "project" && (
        <div className={styles.projectSection}>
          <div className={styles.projectHeader}>
            <h2>Hybrid Edge-Cloud Architecture for AI Surveillance</h2>
            <p className={styles.projectSubtitle}>Using TinyML in Edge Devices for Facial Recognition</p>
          </div>

          <div className={styles.projectCards}>
            <div className={styles.projectCard}>
              <div className={styles.cardIcon}><MdCloud size={40} /></div>
              <h3>Cloud-Edge Hybrid</h3>
              <p>Our architecture combines the power of cloud computing with the speed of edge processing, enabling real-time facial recognition while maintaining privacy and reducing bandwidth requirements.</p>
            </div>
            
            <div className={styles.projectCard}>
              <div className={styles.cardIcon}><MdOutlineCode size={40} /></div>
              <h3>TinyML Technology</h3>
              <p>We implement TinyML techniques including model quantization and knowledge distillation to compress sophisticated facial recognition models to run efficiently on resource-constrained edge devices.</p>
            </div>
            
            <div className={styles.projectCard}>
              <div className={styles.cardIcon}><MdDevices size={40} /></div>
              <h3>Edge Devices</h3>
              <p>The system supports multiple edge devices including Arduino Nicla and Raspberry Pi, each optimized with specific model variants to balance performance and power consumption.</p>
            </div>
            
            <div className={styles.projectCard}>
              <div className={styles.cardIcon}><MdFace size={40} /></div>
              <h3>Facial Recognition</h3>
              <p>Advanced facial recognition algorithms detect and identify individuals with high accuracy while handling challenges such as varying lighting conditions and partial occlusions.</p>
            </div>
          </div>

          <div className={styles.projectDescription}>
            <h3>About the Project</h3>
            <p>
              This project implements a cutting-edge hybrid architecture that combines cloud computing capabilities with edge device processing for facial recognition in AI surveillance applications. By deploying optimized TinyML models on edge devices like Arduino Nicla and Raspberry Pi, we achieve real-time facial recognition while minimizing latency and reducing bandwidth usage.
            </p>
            <p>
              The system processes initial detection on the edge, sending only relevant data to the cloud for further analysis and storage. This hybrid approach ensures privacy by keeping sensitive data local when possible, while leveraging cloud resources for more complex tasks and data aggregation.
            </p>
            <p>
              Our model compression techniques include quantization (reducing numerical precision) and knowledge distillation (training smaller networks to mimic larger ones), allowing sophisticated AI capabilities to run on devices with limited computational resources.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default HelpPage;
