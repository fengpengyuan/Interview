package com.leetcode2;

import java.util.HashMap;

public class LRUCache {
	int capacity;
    HashMap<Integer, CacheNode> map;
    CacheNode head;
    CacheNode tail;
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
        this.capacity=capacity;
        this.head=new CacheNode(-1,-1);
        this.tail=new CacheNode(-1,-1);
        this.map=new HashMap<Integer, CacheNode>();
        tail.pre=head;
        head.next=tail;
    }
    
    public int get(int key) {
        int value=-1;
        if(map.containsKey(key)){
            CacheNode node=map.get(key);
            value=node.val;
            moveToHead(node);
        }
        return value;
    }
    
    public void moveToHead(CacheNode node){
        node.pre.next=node.next;
        node.next.pre=node.pre;
        
        head.next.pre=node;
        node.next=head.next;
        head.next=node;
        node.pre=head;
    }
    
    public void set(int key, int value) {
        if(map.containsKey(key)){
            CacheNode cacheNode=map.get(key);
            cacheNode.val=value;
            moveToHead(cacheNode);
            return;
        }
        if(map.size()==capacity){
            map.remove(tail.pre.key);
            tail.pre=tail.pre.pre;
            tail.pre.next=tail;
        }
        CacheNode node=new CacheNode(key, value);
        node.next=head.next;
        head.next.pre=node;
        head.next=node;
        node.pre=head;
        map.put(key, node);
    }
}
