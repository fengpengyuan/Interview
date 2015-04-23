package com.lintcode;

import java.util.HashMap;

public class LRUCache {
	class Node{
		Node pre;
		Node next;
		int key;
		int val;
		
		public Node(int key, int val){
			this.key=key;
			this.val=val;
		}
	}
	int capacity;
	HashMap<Integer, Node> map=new HashMap<Integer, Node>();
	Node head=new Node(-1,-1);
	Node tail=new Node(-1,-1);
	
	public LRUCache(int capacity){
		this.capacity=capacity;
		head.next=tail;
		tail.next=head;
	}
	
	public int get(int key){
		if(!map.containsKey(key))
			return -1;
		Node node=map.get(key);
		node.pre.next=node.next;
		node.next.pre=node.pre;
		moveToHead(node);
		return node.val;
	}
	
	public void moveToHead(Node node){
		node.next=head.next;
		head.next.pre=node;
		node.pre=head;
		head.next=node;
	}
	
	public void set(int key, int value){
		if(map.containsKey(key)){
			get(key);
			map.get(key).val=value;
			return;
		}
		Node node=new Node(key, value);
		if(map.size()==capacity){
			map.remove(tail.pre.key);
			tail.pre=tail.pre.pre;
			tail.pre.next=tail;
		}
		moveToHead(node);
		map.put(key, node);
	}
}
