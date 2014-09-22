/*!
 * jQuery spriteOnHover Plugin
 * Examples and documentation at: http://apolinariopassos.com.br/dev/spriteonhover
 * Copyright (c) 2012
 * Author: João Paulo Apolinário Passos
 * Version: 0.2.5
 * Licenced under the DWTFYWT public license.
 * http://apolinariopassos.com.br/dev/license
 * Requires: jQuery v1.3.1 or later
 */

(function( $ ){
$.fn.spriteOnHover = function(args){
                     var args = $.extend( {
                      'fps'         : 30,
                      'orientation' : 'horizontal',
                      'rewind' : true,
                      'loop': false,
                      'autostart': false,
                      'repeat': true
                    }, args);

                    var fps = args.fps;
                    var sprite_orientation = args.orientation;
                    var rewind = args.rewind;
                    var loop = args.loop;
                    var autostart = args.autostart;
                    var repeat = args.repeat;
                    if(rewind != false && rewind != 'unanimate'){
                        rewind = true;
                    }
                    if(sprite_orientation != 'vertical' || sprite_orientation === undefined){
                        sprite_orientation = 'horizontal';
                    }

                    var looper_in;
                    var looper_out;
                    var path = $(this).css('background-image').replace('url', '').replace('(', '').replace(')', '').replace('"', '').replace('"', '')
                    var count = $('img[id^="tempImg"]').length + 1;
                    var tempImg = '<img style="max-width: none !important;width: auto!important;min-width: none!important;max-height:none !important;height: auto !important;min-height:none !important" id="tempImg'+count+'" src="' + path + ' "/>';

                    $('body').append(tempImg);
                    $('#tempImg'+count).hide();

                    var frame_width = $(this).width();
                    var frame_height = $(this).height();
                    var this_parent = $(this);

                    $('#tempImg'+count).load(function(){
                            var executou = false;
                            if(sprite_orientation == 'horizontal'){
                                var frame_size = $(this_parent).width();
                                var real_size = $(this).width();
                            }
                            else{
                                var real_size = $(this).height();
                                var frame_size = $(this_parent).height();
                            }
                            var number_of_frames = (real_size/frame_size);
                            var margin_size = frame_size;
                            var executando = false;
                            function withMouseIn(){
                                    if(!executou){
                                        margin_size = frame_size;
                                        var counter = 1;
                                        var mouseleft = false;
                                        var finalizouAnimacao = false;
                                        looper_in = setInterval(function(){
                                                counter++;
                                                executando = true;
                                                    $(this_parent).mouseleave(function(){
                                                        if(loop != 'infinite'){
                                                            counter = number_of_frames;
                                                            mouseleft = true;
                                                            if(!finalizouAnimacao){
                                                                if(!repeat)
                                                                executou = true;
                                                            }
                                                        }
                                                    });
                                                    if(repeat){
                                                        var backgroundPos = $(this_parent).css('background-position').split(" ");
                                                        if(sprite_orientation == 'horizontal')
                                                            var Pos = backgroundPos[0];
                                                        else
                                                            var	Pos = backgroundPos[1];
                                                        if(parseInt(Pos)*-1 == frame_size*(number_of_frames-1) && mouseleft){
                                                            clearInterval(looper_in);
                                                            executando = false;
                                                            executou = false;
                                                            finalizouAnimacao = true;
                                                        }
                                                    }
                                                if(counter == (number_of_frames)){
                                                            if(loop==true || loop == 'infinite'){
                                                                $(this_parent).css("background-position","-200px 0px");
                                                                margin_size = frame_size;
                                                                counter = 0;
                                                            }
                                                            else{
                                                                clearInterval(looper_in);
                                                                executando = false;
                                                            }
                                                            if(!repeat)
                                                                executou = true;
                                                }
                                                else{
                                                        if(margin_size <= (real_size-(frame_size)))
                                                        if(sprite_orientation == 'horizontal'){
                                                            $(this_parent).css("background-position","-"+margin_size+"px 0px");
                                                        }else{
                                                            $(this_parent).css("background-position","0px -"+margin_size+"px");
                                                        }
                                                        margin_size = margin_size+frame_size;
                                                }
                                            },(parseInt(1000/fps)));
                                    }

                            }
                            function withMouseOut(){
                                if(rewind == true){
                                        var counter = 1;
                                        looper_out = setInterval(function(){
                                            counter++;
                                            var backgroundPos = $(this_parent).css('background-position').split(" ");
                                            if(sprite_orientation == 'horizontal')
                                                var Pos = backgroundPos[0];
                                            else
                                                var	Pos = backgroundPos[1];
                                            if(counter == (number_of_frames) || parseInt(Pos) == 0){
                                                executando = false;
                                                if(repeat)
                                                    executou = false;
                                                $(this_parent).css("background-position","-200px 0px");
                                                clearInterval(looper_out);
                                            }
                                            margin_size = margin_size-frame_size;
                                            if(margin_size <= (real_size-(frame_size))){
                                                if(sprite_orientation == 'horizontal')
                                                    $(this_parent).css("background-position","-"+margin_size+"px 0px");
                                                else
                                                    $(this_parent).css("background-position","-200px -"+margin_size+"px");
                                            }
                                        },(parseInt(1000/fps)));
                                }else{
                                    executou = true;
                                    executando = false;
                                }
                            }

                            var executou = false;
                            if(autostart == true){
                                withMouseIn();
                                if(loop != 'infinite'){
                                    autostart = false;
                                }
                            }
                            $(this_parent).hover(
                                function(){
                                        if(looper_out != undefined){
                                            clearInterval(looper_out);
                                        }
                                        if(!autostart){
                                            if(!executando)
                                                withMouseIn();
                                        }
                                    },
                                function(){
                                        if(loop == false && rewind==true){
                                            withMouseOut();
                                            if(repeat)
                                                executou = false;
                                            clearInterval(looper_in);
                                        }
                                        else if(loop != 'infinite'){
                                            if(rewind == 'unanimate'){
                                                $(this_parent).css("background-position","-200px 0px");
                                                clearInterval(looper_in);
                                                if(repeat)
                                                    executou = false;
                                                executando = false;
                                            }
                                            else if(!autostart){
                                                if(rewind==true && loop != 'infinite'){
                                                    withMouseOut();
                                                    if(repeat)
                                                        executou = false;
                                                    clearInterval(looper_in);
                                                }else if(rewind==false && loop != 'infinite'){
                                                    if(repeat)
                                                        executou = false
                                                }
                                            }
                                        }
                                    }
                                );
                            $(this).remove();
                    });
                }
})( jQuery );
