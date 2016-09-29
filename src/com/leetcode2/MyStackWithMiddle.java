package com.leetcode2;

public class MyStackWithMiddle {
	DListNode head;
	DListNode mid;
	int count;
	
	public MyStackWithMiddle(){
		head=null;
		mid=null;
		count=0;
	}
	
	public void push(int data){
		if(head==null){
			head=new DListNode(data);
			mid=head;
			count++;
		}else{
			DListNode node=new DListNode(data);
			node.next=head;
			head.pre=node;
			head=node;
			count++;
			if(count%2==1){
				mid=mid.pre;
			}
		}
	}
	
	public int pop(){
		if(count==0){
			return -1;
		}
		int res = head.val;
		head=head.next;
		if(head!=null)
			head.pre=null;
		count--;
		if(count%2==0)
			mid=mid.next;
		return res;
	}
	
	public int findMid(){
		if(mid!=null)
			return mid.val;
		return -1;
	}
	
	
	public static void main(String[] args){
		MyStackWithMiddle stk=new MyStackWithMiddle();
		System.out.println(stk.findMid());
		stk.push(10);
		System.out.println(stk.findMid());
		System.out.println(stk.pop());
		stk.push(11);
		stk.push(12);
		System.out.println(stk.findMid());
		stk.push(13);
		System.out.println(stk.findMid());
		System.out.println(stk.pop());
		System.out.println(stk.findMid());
		System.out.println(stk.pop());
		System.out.println(stk.findMid());
		System.out.println(stk.pop());
		System.out.println(stk.findMid());
		
		System.out.println();
		stk.push(100);
		System.out.println(stk.findMid());
		stk.push(101);
		System.out.println(stk.findMid());
		stk.push(102);
		System.out.println(stk.findMid());
		stk.push(103);
		System.out.println(stk.findMid());
		stk.push(104);
		System.out.println(stk.findMid());
		stk.push(105);
		System.out.println(stk.findMid());
		
		System.out.println();
		
		System.out.println(stk.pop());
		System.out.println(stk.findMid());
		System.out.println(stk.pop());
		System.out.println(stk.findMid());
		System.out.println(stk.pop());
		System.out.println(stk.findMid());
		System.out.println(stk.pop());
		System.out.println(stk.findMid());
		
		System.out.println(stk.pop());
		System.out.println(stk.findMid());
		System.out.println("size "+stk.count);
		stk.pop();
		System.out.println();
		
		
		stk.push(11);
		stk.push(22);
		stk.push(33);
		stk.push(44);
		stk.push(55);
		stk.push(66);
		stk.push(77);
		
		System.out.println(stk.pop());
		System.out.println(stk.findMid());
		System.out.println(stk.pop());
		System.out.println(stk.findMid());
		
	}

}
