package com.leetcode2;

import java.util.LinkedList;
import java.util.Queue;

public class MyStack {
	Queue<Integer> q=new LinkedList<Integer>();
    // Push element x onto stack.
    public void push(int x) {
        q.add(x);
        int n=q.size();
        while(n>1){
            q.add(q.remove());
            n--;
        }
    }

    // Removes the element on top of the stack.
    public void pop() {
        q.remove();
    }

    // Get the top element.
    public int top() {
        return q.peek();
    }

    
   // using two queues
    
//    // Return whether the stack is empty.
//    public boolean empty() {
//        return q.isEmpty();
//    }
//    
//    
//    
//    Queue<Integer> q1=new LinkedList<Integer>();
//    Queue<Integer> q2=new LinkedList<Integer>();
//    // Push element x onto stack.
//    public void push(int x) {
//        q1.offer(x);
//    }
//
//    // Removes the element on top of the stack.
//    public void pop() {
//        top();
//        q1.poll();
//        Queue q=q1;
//        q1=q2;
//        q2=q;
//    }
//
//    // Get the top element.
//    public int top() {
//        while(q1.size()>1){
//            q2.offer(q1.poll());
//        }
//        return q1.peek();
//    }
//
//    // Return whether the stack is empty.
//    public boolean empty() {
//        return q1.isEmpty() && q2.isEmpty();
//    }
}
