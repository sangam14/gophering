(window.webpackJsonp=window.webpackJsonp||[]).push([[36],{"9MUH":function(e,t,a){},BJHg:function(e,t,a){"use strict";a.r(t);var r=a("q1tI"),n=a.n(r),l=a("Wbzz"),c=a("VXBa"),i=a("FwLO"),s=a("R3Jx"),o=a("Tp9X"),d=a.n(o),u=a("YBaW"),m=(a("SI/r"),a("9MUH"),a("LF3+")),p=a.n(m);t.default=function(e){var t=e.data,a=e.location,o=t.ghostPost,m=t.all_posts.edges,f=Object(u.a)(o,m);return Object(r.useEffect)((function(){d()()}),[]),n.a.createElement(c.a,null,n.a.createElement(i.a,{location:a,data:t,type:"article"}),n.a.createElement("div",{className:"post-view post-view--resources"},n.a.createElement("article",{className:"wrapper-800"+(null==o.feature_image?" no-feature-image":"")},n.a.createElement("header",{className:"post__header"},n.a.createElement("div",{className:"post__primary-tag"},n.a.createElement(l.Link,{to:"/resources/",className:"go-back-link"},n.a.createElement("span",{className:"icon"},n.a.createElement(p.a,null)),n.a.createElement("span",{className:"text"},"Resources"))),n.a.createElement("h1",{className:"post__title"},o.title)),n.a.createElement("div",{className:"post__body"},n.a.createElement("div",{className:"load-external-scripts",dangerouslySetInnerHTML:{__html:o.codeinjection_head}}),n.a.createElement("div",{className:"post-full-content load-external-scripts",dangerouslySetInnerHTML:{__html:o.html}}))),f.length>0&&n.a.createElement("div",{className:"related-posts related-posts--resources"},n.a.createElement("div",{className:"wrapper-1200"},n.a.createElement("div",{className:"related-posts__header"},"Related Resources"),n.a.createElement("div",{className:"post-feed"},f.map((function(e){return n.a.createElement(s.a,{data:e.node,key:e.node.id})}))))),n.a.createElement("div",{className:"load-external-scripts",dangerouslySetInnerHTML:{__html:o.codeinjection_foot}})))}},"LF3+":function(e,t,a){var r=a("q1tI");function n(e){return r.createElement("svg",e,r.createElement("g",{fill:"none",fillRule:"evenodd"},[r.createElement("path",{d:"M0 0h24v24H0z",key:0}),r.createElement("path",{fill:"#1E54D5",d:"M16.707 7.293l-1.414 1.414L17.586 11H2v2h15.586l-2.293 2.293 1.414 1.413L21.414 12z",key:1})]))}n.defaultProps={width:"24",height:"24",viewBox:"0 0 24 24"},e.exports=n,n.default=n},P1fb:function(e,t,a){"use strict";t.a=function(e){var t=null==e?void 0:e.match(/(?:(?:https?|ftp|file):\/\/|www\.|ftp\.)(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[-A-Z0-9+&@#\/%=~_|$?!:,.])*(?:\([-A-Z0-9+&@#\/%=~_|$?!:,.]*\)|[A-Z0-9+&@#\/%=~_|$])/gi);return t?t[0]:null}},R3Jx:function(e,t,a){"use strict";var r=a("q1tI"),n=a.n(r),l=a("Wbzz"),c=(a("l4yw"),a("P1fb")),i=function(e){var t,a,r,i,s=e.data,o="Learn more";if(s.tags.find((function(e){return"hash-private"===e.slug})))o="Request "+s.tags[1].name;else switch(s.tags[1].slug){case"white-paper":o="Read white paper";break;case"datasheet":o="Read datasheet";break;case"case-study":o="Read case study";break;case"webinar":o="Watch webinar";break;case"video":o="Watch video";break;default:o="Read more"}var d=(null==s||null===(t=s.featureImageSharp)||void 0===t||null===(a=t.childImageSharp)||void 0===a||null===(r=a.fluid)||void 0===r?void 0:r.src)||(null==s||null===(i=s.featureImageSharp)||void 0===i?void 0:i.publicURL)||(null==s?void 0:s.feature_image),u=Object(c.a)(null==s?void 0:s.codeinjection_head);return n.a.createElement("div",{className:"resource-card resource-card--"+s.tags[1].slug+" post-feed__card"},u&&n.a.createElement("a",{href:u,className:"resource-card__wrapper"},n.a.createElement("div",{className:"resource-card__image"},n.a.createElement("img",{src:d,alt:s.title})),n.a.createElement("div",{className:"resource-card__content"},n.a.createElement("div",{className:"resource-card__tag"},s.tags[1].name),n.a.createElement("h2",{className:"resource-card__title"},s.title),n.a.createElement("div",{className:"link--with-arrow"},o))),!u&&n.a.createElement(l.Link,{to:"/"+s.primary_tag.slug+"/"+s.slug+"/",className:"resource-card__wrapper"},n.a.createElement("div",{className:"resource-card__image"},n.a.createElement("img",{src:d,alt:s.title})),n.a.createElement("div",{className:"resource-card__content"},n.a.createElement("div",{className:"resource-card__tag"},s.tags[1].name),n.a.createElement("h2",{className:"resource-card__title"},s.title),n.a.createElement("div",{className:"link--with-arrow"},o))))};t.a=i},"SI/r":function(e,t,a){},Tp9X:function(e,t){var a=['iframe[src*="player.vimeo.com"]','iframe[src*="youtube.com"]','iframe[src*="youtube-nocookie.com"]','iframe[src*="kickstarter.com"][src*="video.html"]',"object"];function r(e,t){return"string"==typeof e&&(t=e,e=document),Array.prototype.slice.call(e.querySelectorAll(t))}function n(e){return"string"==typeof e?e.split(",").map(c).filter(l):function(e){return"[object Array]"===Object.prototype.toString.call(e)}(e)?function(e){return[].concat.apply([],e)}(e.map(n).filter(l)):e||[]}function l(e){return e.length>0}function c(e){return e.replace(/^\s+|\s+$/g,"")}e.exports=function(e,t){var c;t=t||{},c=e=e||"body","[object Object]"===Object.prototype.toString.call(c)&&(t=e,e="body"),t.ignore=t.ignore||"",t.players=t.players||"";var i=r(e);if(l(i)){var s;if(!document.getElementById("fit-vids-style"))(document.head||document.getElementsByTagName("head")[0]).appendChild(((s=document.createElement("div")).innerHTML='<p>x</p><style id="fit-vids-style">.fluid-width-video-wrapper{width:100%;position:relative;padding:0;}.fluid-width-video-wrapper iframe,.fluid-width-video-wrapper object,.fluid-width-video-wrapper embed {position:absolute;top:0;left:0;width:100%;height:100%;}</style>',s.childNodes[1]));var o=n(t.players),d=n(t.ignore),u=d.length>0?d.join():null,m=a.concat(o).join();l(m)&&i.forEach((function(e){r(e,m).forEach((function(e){u&&e.matches(u)||function(e){if(/fluid-width-video-wrapper/.test(e.parentNode.className))return;var t=parseInt(e.getAttribute("width"),10),a=parseInt(e.getAttribute("height"),10),r=isNaN(t)?e.clientWidth:t,n=(isNaN(a)?e.clientHeight:a)/r;e.removeAttribute("width"),e.removeAttribute("height");var l=document.createElement("div");e.parentNode.insertBefore(l,e),l.className="fluid-width-video-wrapper",l.style.paddingTop=100*n+"%",l.appendChild(e)}(e)}))}))}}},YBaW:function(e,t,a){"use strict";var r=a("LvDl"),n=function(e,t){var a=new Date(e.node.published_at).getTime(),r=new Date(t.node.published_at).getTime();return a>r?-1:a<r?1:0};t.a=function(e,t){var a=[],l=r.filter(t,(function(t){var n=t.node;if(e.slug===n.slug)return!1;var l=r.intersectionBy(e.tags,n.tags,(function(e){return e.slug}));return l.length>2&&a.push({slug:n.slug,tags:l.length}),l.length>=2}));if(l.length&&a.length){var c=[];a=r.sortBy(a,"tags"),r.forEach(a,(function(e){c.push(r.find(l,(function(t){return t.node.slug===e.slug})))})),l=r.difference(l,c),l=r.concat(c,l.sort(n)).slice(0,3)}else l.length&&(l.length>3?l=l.sort(n).slice(0,3):l.length>1&&(l=l.sort(n)));if(l.length<3){var i=3-l.length;t=r.filter(t,(function(t){return t.node.slug!==e.slug}));for(var s=r.difference(t,l),o=[];i>0&&s.length;){var d=Math.floor(Math.random()*(s.length+1)),u=s.splice(d,1)[0];u&&(o.push(u),i-=1)}return r.concat(l,o)}return l}},l4yw:function(e,t,a){}}]);
//# sourceMappingURL=component---src-templates-resource-post-tsx-4ac06591ff4710660893.js.map