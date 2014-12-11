package linkedIn;

public class Relation {
	public Integer parent;
    public Integer child;
    public boolean isLeft;
    public Relation(Integer child, Integer parent, boolean isLeft){
    	this.child=child;
    	this.parent=parent;
    	this.isLeft=isLeft;
    }
}
