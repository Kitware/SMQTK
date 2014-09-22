//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
// Container class


function ResultsContainer(divid) 
    {
    this.Init(divid);
    }

ResultsContainer.prototype.Init = function(divid)
    {
    // Create defaults
    this.results = [];
    this.w = 100;
    this.h = 100;
    // Create sims
    this.results = [];
    // Initialize container div
    this.div =$("#" + divid); 
    //$(this.div).sortable();
    //
    //$(this.div).draggable(
    //    { grid: 100,}
    //);
    $(this.div).css({"padding" : "4px"});
    }

ResultsContainer.prototype.OnQueryClose = function(div)
    {
    var parent = $(div).parent();
    var idx = this.results.indexOf($(div).parent());
    this.results.splice(idx,1);    
    $(parent).remove();     
    }

ResultsContainer.prototype.Remove = function(node, queriesToo, self)
    {
    console.log("Processing node: ", node.data.title);    
    // By default delete associated queries too
    queriesToo = queriesToo || true;
    self = self || false;

    if(queriesToo=== true)
        {
        var children = node.getChildren();            
        if(children)
            {
            // Recursively Remove children
            for(var i = 0; i < children.length; i++)
                {
                this.Remove(children[i], true, false)
                console.log("Next" + i)
                }
           }
        else
            {
            console.log("No chilren for " + node.data.title);
            }
        // Then remove if query div is associated with this node
        if(node.data.editCtrl)
            {
            var idx = this.results.indexOf(node.data.editCtrl.div);
            if(idx == null)
                {
                console.log("editctrl div not found ")
                }
            else
                {
                this.results.splice(idx,1);
                $(node.data.editCtrl.div).remove();
                }
            }
        else
            {
            console.log("editctrl not found ")
            }
        }
    if(self)
        {    
        console.log("Removing node: " + node.data.title);
        node.remove();    
        }
    }

ResultsContainer.prototype.Reset = function(timer) 
    {
    for (var i = 0; i < this.results.length; i++) 
        {
        // Do something
        };
        return 0;
    }


ResultsContainer.prototype.AddQuery = function(query) 
    {
    // query is an object
    // Root node, tree with attributes
    // Containes everything needed for editing and initializing the query
    // Create new result based on parameters and add it to the results
    var newresult;  
    if(query.node.data.isFolder === false)
        {
        newresult = new AttributeQueryResults(this.div, query)
        }
    else
        {
        newresult = new ExpressionQueryResults(this.div, query)
        }
    this.results.push(newresult);
    return(newresult);        
    }
