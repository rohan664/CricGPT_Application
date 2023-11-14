import "./App.css";
import logo from "./assets/crickrt_logo.svg";
import saved from "./assets/bookmark.svg";
import send from "./assets/send.svg";
import user from "./assets/user-icon.png";
import menu from "./assets/menu.svg";
import cancel from "./assets/cancel.svg";
import React, { useEffect, useRef } from "react";
import { useState } from "react";
import axios from "axios";

function App() {
  const msgEnd = useRef(null);

  const [chathistory, setChathistory] = useState([]);
  const [message, setMessage] = useState("");
  const [menuicon, setMenuicon] = useState(true);
  const [currentChat, setCurrentChat] = useState("");
  const [fetchChatHistory , setfetchChatHistory] = useState("")
  const [chat, setChat] = useState([
    {
      text: "Welcome to CricGPT World",
      isBot: true,
    },
  ]);

  function addChatToHistory() {
    let value = document.getElementById("search-box").value;
    setMessage(value);
    if (chat[0].text === "Welcome to CricGPT World") {
      let newchat = chat.splice(0, 1);
      setChat([
        ...chat,
        { text: value, isBot: false },
        { text: "Generating...", isBot: true },
      ]);
      if (localStorage.getItem("chat") === null && localStorage.getItem("chatHistory") === null) {
        setCurrentChat(value);
        let list = [value];
        localStorage.setItem("chat", JSON.stringify(list));

        let chatList = {};
        chatList[value] = [{ text: value, isBot: false }];
        localStorage.setItem("chatHistory", JSON.stringify(chatList));
      } 
      else {
        // this is chat history
        setCurrentChat(value);
        const existingListString = localStorage.getItem("chat");
        const newItem = value;
        let updatedList = [];
        if (existingListString) {
          updatedList = JSON.parse(existingListString);
        }
        updatedList.push(newItem);
        localStorage.setItem("chat", JSON.stringify(updatedList));

        // --------------------------------
        const getChatHistoryDetails = localStorage.getItem("chatHistory");
        let updatedChatHistoryDetails = {};
        updatedChatHistoryDetails = JSON.parse(getChatHistoryDetails);
        updatedChatHistoryDetails[value] = [{ text: value, isBot: false }];
        localStorage.setItem("chatHistory", JSON.stringify(updatedChatHistoryDetails));


      }
      let chathis = localStorage.getItem("chat");
      if (chathis) {
        let maintainChatHistory = JSON.parse(chathis).map((value) => {
          // let parsedChatHistory = JSON.parse(chathis);
          setChathistory([...chathistory,value]);
        });
    }
    } 
    else {
      const getChatHistoryDetails = localStorage.getItem("chatHistory");
      let updatedChatHistoryDetails = {};
      updatedChatHistoryDetails = JSON.parse(getChatHistoryDetails);
      Object.keys(updatedChatHistoryDetails).forEach((key) => {
        if (key === currentChat) {
          updatedChatHistoryDetails[key].push({text: value, isBot: false })
          localStorage.setItem("chatHistory",JSON.stringify(updatedChatHistoryDetails))
        }
      });
      setChat([
        ...chat,
        { text: value, isBot: false },
        { text: "Generating...", isBot: true },
      ]);
    }
    document.getElementById("search-box").value = "";
  }

  function handleKeyDown(e) {
    if (e.key === "Enter") {
      addChatToHistory();
    }
  }

  useEffect(() => {
    let chathis = localStorage.getItem("chat");
    let li = []
    if (chathis) {
      let maintainChatHistory = JSON.parse(chathis).map((value) => {
        let parsedChatHistory = JSON.parse(chathis);
        setChathistory((prevHistory) => [...parsedChatHistory]);
      });
    }
  },[]);

  useEffect(() => {
    msgEnd.current.scrollIntoView();
  }, [chat]);

  useEffect(() => {
    const apiUrl = "https://cricgpt-n6lr75fo6q-el.a.run.app/chat";

    const requestBody = {
      message: message,
    };

    axios
      .post(apiUrl, requestBody)
      .then((response) => {
        let removelastObject = chat.splice(chat.length - 1, 1);
        const getChatHistoryDetails = localStorage.getItem("chatHistory");
        let updatedChatHistoryDetails = {};
        updatedChatHistoryDetails = JSON.parse(getChatHistoryDetails);
        Object.keys(updatedChatHistoryDetails).forEach((key) => {
          if (key === currentChat) {
            updatedChatHistoryDetails[key].push({text: response.data.response, isBot: true })
            localStorage.setItem("chatHistory",JSON.stringify(updatedChatHistoryDetails))
          }
        });
        setChat([...chat, { text: response.data.response, isBot: true }]);
      })
      .catch((error) => {
        console.error("Error fetching data:", error);
      });
  }, [message]);

  function handleNewchat() {
    window.location.reload();
  }

  function menuClickEvent() {
    if (menuicon) {
      document.getElementById("menu").style.display = "block";
      setMenuicon(false);
    } else {
      document.getElementById("menu").style.display = "none";
      setMenuicon(true);
    }
  }

  function closeSideBar() {
      document.getElementById("menu").style.display = "none";
      setMenuicon(true);
    
  }

  function chatHistoryRecord(value){
    const getChatHistoryDetails = localStorage.getItem("chatHistory");
    Object.keys(JSON.parse(getChatHistoryDetails)).forEach((key) => {
      if (key === value) {
        setChat(JSON.parse(getChatHistoryDetails)[key])
      }
    });
  }

  // useEffect(()=>{
  //   chatHistoryRecord(fetchChatHistory)
  // },[fetchChatHistory])

  return (
    <div className="App">
      <div className="sideBar" id="menu">
        <div className="upperSide">
          <div className="cancel-icon" onClick={closeSideBar}>
            <img src={cancel} alt="logo" className="logo" />
          </div>
          <div className="uppersideTop">
            <img src={logo} alt="logo" className="logo" />
            <span className="title">CRICGPT</span>
          </div>
        </div>
        <div className="middleSide">
          <div className="chat-history">Chat History</div>
          {/* <img src={playerInformation.rohit} alt='' className='player-profile'/><span className='name'>{'Virat Kohli'}</span> */}
        </div>
        <div className="lowerSide">
          {chathistory.map((value, index) => {
            return (
              <div className="query" key={index}  onClick={()=>{chatHistoryRecord(value)}}>
                <img src={saved} alt="logo" />
                <span className="bookmark-message">{value}</span>
              </div>
            );
          })}
        </div>
        <div className="new-chat">
          <div className="new-chat-button" onClick={handleNewchat}>
            New Chat
          </div>
        </div>
      </div>
      <div className="main">
        <div className="menu-icon">
          <img src={menu} alt="menu-icon" onClick={menuClickEvent}></img>
        </div>
        <div className="chats">
          {chat.map((chat, i) => {
            return (
              <div className={chat.isBot ? "chat bot" : "chat"} key={i}>
                <img
                  src={chat.isBot ? logo : user}
                  alt=""
                  className="chatimg"
                />
                <p className="text">{chat.text}</p>
              </div>
            );
          })}
          <div ref={msgEnd} />
        </div>
        <div className="chatFooter">
          <div className="inp">
            <input
              type="text"
              name=""
              id="search-box"
              placeholder="search here ....."
              onKeyDown={handleKeyDown}
            />
            <button className="send">
              <img
                src={send}
                alt="send"
                className="send"
                onClick={addChatToHistory}
              />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
