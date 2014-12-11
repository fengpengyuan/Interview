package com.leetcode;

import java.util.HashMap;
import java.util.Iterator;

public class LRUCache {
    int capacity;
    HashMap<Integer,CacheNode> map = new HashMap<Integer,CacheNode>();
    CacheNode head=null;
    CacheNode tail=null;
    
    class CacheNode{
        int key;
        int val;
        CacheNode pre;
        CacheNode next;
        public CacheNode(int key, int val){
            this.key=key;
            this.val=val;
        }
    }
    public LRUCache(int capacity) {
        this.capacity = capacity;
    }
    
    public int get(int key) {
        if(!map.containsKey(key))
            return -1;
        CacheNode node=map.get(key);
        CacheNode pre=node.pre;
        CacheNode next=node.next;
        if(pre!=null){
            pre.next=next;
            if(next!=null)
                next.pre=pre;
            else
                tail=pre;
            
            node.next=head;
            head.pre=node;
            node.pre=null;
            head=node;
        }
        return node.val;
    }
    
    public void set(int key, int value) {
        if(map.containsKey(key)){
            get(key); // just get to the head
            map.get(key).val=value;
        }
        else{
            CacheNode node=new CacheNode(key, value);
            if(map.size()==capacity){
                if(tail!=null){
                    CacheNode old=tail;
                    tail=old.pre;
                    if(tail!=null)
                        tail.next=null;
                    map.remove(old.key);
                }
            }
            
            if(head==null){
                head=node;
                tail=node;
            }
            else{
                node.next=head;
                head.pre=node;
                node.pre=null;
                head=node;
            }
            map.put(key,node);
        }
    }
    
    public static void main(String[] args){
    	LRUCache cache= new LRUCache(2);
    	cache.set(2, 1);
    	cache.set(2,2);
    	HashMap<Integer,CacheNode> map=cache.map;
    	Iterator<Integer> it=map.keySet().iterator();
    	while(it.hasNext()){
    		int key=it.next();
    		System.out.println(key+" "+map.get(key).val);
    	}
    	System.out.println(cache.get(2));
    	cache.set(1,1);
    	it=map.keySet().iterator();
    	while(it.hasNext()){
    		int key=it.next();
    		System.out.println(key+" "+map.get(key).val);
    	}
    	
    	cache.set(4,1);
    	
    	it=map.keySet().iterator();
    	while(it.hasNext()){
    		int key=it.next();
    		System.out.println(key+" "+map.get(key).val);
    	}
    	System.out.println(map.size());
    	System.out.println(cache.capacity);
    	System.out.println(cache.get(2));
    	
    }


	
}
