package com.leetcode;

import java.util.LinkedList;
import java.util.Queue;

public class TrieNode {
	char c;
	boolean isComplete;
	TrieNode[] children;
	
	static TrieNode root;
	public TrieNode createNewNode(char c){
		TrieNode node=new TrieNode();
		node.c=c;
		node.isComplete=false;
		node.children=new TrieNode[26];
		
		return node;
	}
	
	public void initialize(){
		root=createNewNode(' ');
	}
	
	public boolean search(String s){
		int len=s.length();
		if(len==0)
			return true;
		TrieNode ptr=root;
		int i=0;
		while(i<len){
			if(ptr.children[s.charAt(i)-'a']==null)
				break;
			ptr=ptr.children[s.charAt(i)-'a'];
			i++;
		}
		return i==len&&ptr.isComplete;
	}
	
	public void insert(String s){
		TrieNode node=root;
		for(int i=0;i<s.length();i++){
			if(node.children[s.charAt(i)-'a']==null){
				node.children[s.charAt(i)-'a']=createNewNode(s.charAt(i));
			}
			node=node.children[s.charAt(i)-'a'];
		}
		node.isComplete=true;
	}
	
	public void printTire(TrieNode root){
		if(root==null)
			return;
		Queue<TrieNode> que=new LinkedList<TrieNode>();
		int curlevel=0;
		int nextlevel=0;
		que.add(root);
		curlevel++;
		while(!que.isEmpty()){
			TrieNode node=que.remove();
			curlevel--;
			System.out.print(node.c);
			for(int i=0;i<26;i++){
				if(node.children[i]!=null){
					que.add(node.children[i]);
					nextlevel++;
				}
			}
			if(curlevel==0){
				System.out.println();
				curlevel=nextlevel;
				nextlevel=0;
			}			
		}
	}
	
	public static void main(String[] args){
		TrieNode trie=new TrieNode();
		trie.initialize();
		
		trie.insert("tree");
		trie.insert("trie");
		trie.insert("trade");
		
		trie.printTire(root);
		
	}
}
