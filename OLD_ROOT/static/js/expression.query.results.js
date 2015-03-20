//
// LICENCE
// -------
// Copyright 2013 by Kitware, Inc. All Rights Reserved. Please refer to
// KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
// Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
//
// Inheatrits from lgoquery results
// Adds specifics for attribute node editing and results display

// Inherit from Queryresults

ExpressionQueryResults.prototype = Object.create(QueryResults.prototype);
ExpressionQueryResults.prototype.constructor = ExpressionQueryResults;


var get_first_leaf = function(node)
    {
    return '52';
    }


var add_sample_tree = function()
    {
    var node = $("#tree").dynatree("getTree").getRoot();
    var landing = node.addChild({'title': "Fish Landing", 'isFolder' : true});
    landing.addChild({'title': "$and", "isFolder" : true, "children" :
            [ {"title" : "natural", min : 0.5, index : '87'},
              {"title" : "natural_light", min : 0.5, index : '82' },
              {"title" : "$or", "isFolder" : true, "children" :
                    [ {"title" : "ocean", min : 0.5, index : '52'},
                      {"title" : "still_water", min : 0.5, index : '54'}
                    ]},
            ]  });

    var woodworking = node.addChild({'title': "Wood Working ", 'isFolder' : true});
    woodworking.addChild({'title': "$and", "isFolder" : true, "children" :
            [
                {"title" : "wood_(not_part_of_a_tree)", min : 0.5, index:'70'},
                {"title" : "working", min : 0.5, index:'31'},
                {"title" : "using_tools", min : 0.5, index:'32'}
            ]});

    var trick = node.addChild({'title': "Board Trick", 'isFolder' : true});
    trick.addChild({'title': "$and", "isFolder" : true, "children" :
            [ {"title" : "sports", min : 0.5, index:'23'},
              {"title" : "$or", "isFolder" : true, "children" :
                    [ {"title" : "asphalt", min : 0.5, index:'60'},
                      {"title" : "pavement", min : 0.5, index:'61'},
                      {"title" : "snow", min : 0.5, index:'56'},
                    ]},
            ]  });

    var test = node.addChild({'title': "Test", 'isFolder' : true, "isOperator" : false});
    test.addChild({'title': "sports", "isFolder" : false, 'min' : 0.5, index: '70'});

    return {'test' : test, 'trick' : trick, 'woodworking' : woodworking}
    }

var tree_json = function (node)
    {
    if(node.getLevel() === 1)
        {
        // Need to get one step down
        var children = node.getChildren();
        return tree_json(children[0])
        }
    var string = "{ ";

    if(node.data.isFolder === true)
        {
        string =string + '"' + node.data.title +'" : [ ';

        var children = node.getChildren();
        if(children === null)
            {
            string = string + "]"
            return string;
            }
        else
            {
            for(var i=0; i < children.length; i ++)
                {
                string = string + " " + tree_json(children[i]);
                if(i < children.length - 1)
                    {
                    string = string + ", ";
                    }
                }
                string = string + "] "
            }
        }
     else
        {
        var min = 0.0; // TODO: Give ui for default threshold
        if(node.data.min !== undefined)
            {
            min = node.data.min;
            }
        string ="{ " + '"' + node.data.title +'" : { "$gte" :' + (node.data.min || 0) + '  }';

        }
    return string + " }";
    }

function ExpressionQueryResults(container, query)
    {
    // Call base class init
    console.log("ExpressionQueryResults")
    this.parent = QueryResults.call( this, container, query);

    // Now assume that basics are created Now add
    this.expression = document.createElement('div');
    $(this.expression).css({"float" : "clear"});
    $(this.expression).html(traverse(this.query.node));
    $(this.expression).appendTo($(this.controls));

    // Assume that calibration exists for the attributes that are going to come
    // Fall back to the known percentile numbers in case calibration not existing

    // Initializee to defaults
    }


ExpressionQueryResults.prototype.UpdateQuery = function()
    {
    // Form the query from known information
    // Create the tree
    var tree_str  = tree_json(this.node);
    var sort = get_first_leaf(this.node);

    this.url = "/zero_shot/query_fusion";
    // Async make the query
    this.query =
        {
        'dataset' : "avg",
        'algo' : this.algo,
        "tree" : encodeURIComponent(tree_str),
        }
    }

