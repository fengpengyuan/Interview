package com.leetcode2;

import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

class Log{
	int timestamp;
    String message;
    public Log(int aTimestamp, String aMessage) {
        timestamp = aTimestamp;
        message = aMessage;
    }
	
}

public class Logger {
//	HashMap<String,Integer> map;
//	/** Initialize your data structure here. */
//	public Logger() {
//	    map=new HashMap<>();
//	}
//
//	/** Returns true if the message should be printed in the given timestamp, otherwise returns false. The timestamp is in seconds granularity. */
//	public boolean shouldPrintMessage(int timestamp, String message) {
//	//update timestamp of the message if the message is coming in for the first time,or the last coming time is earlier than 10 seconds from now
//	    if(!map.containsKey(message)||timestamp-map.get(message)>=10){
//	        map.put(message,timestamp);
//	        return true;
//	    }
//	    return false;
//	}
	
	PriorityQueue<Log> que;
	Set<String> recentMessage;
	
	 public Logger() {
		 que = new PriorityQueue<Log>(10, new Comparator<Log>(){

			@Override
			public int compare(Log log1, Log log2) {
				// TODO Auto-generated method stub
				return log1.timestamp-log2.timestamp;
			}
			 
		 });
		 
		 recentMessage = new HashSet<String>();
	 }
	 
	 public boolean shouldPrintMessage(int timestamp, String message) {
		 while(que.size()>0){
			 Log log = que.peek();
			 if(timestamp - log.timestamp>10){
				 que.poll();
				 recentMessage.remove(log.message);
			 }else
				 break;
		 }
		 
		 if(recentMessage.contains(message))
			 return false;
		 else{
			 que.add(new Log(timestamp, message));
			 recentMessage.add(message);
			 return true;
		 }
	 }
}
